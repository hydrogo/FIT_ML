import os
import time
import pickle
import argparse

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


###################
# Global parameters

# arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("--freq", type=str, default="W", help="Temporal resolution: W for weekly, or M for monthly")
parser.add_argument("--n_trees", type=int, default=10, help="Number of decision trees in Random Forest ensemble: 10, 100")
parser.add_argument("--subset_jjas", type=bool, default=False, help="True for subseting only June, July, August, and September. False for considering the entire timeseries")
parser.add_argument("--only_t2m_mean", type=bool, default=True, help="True for considering only mean temperature. False for considering mean, min, max temperatures")
parser.add_argument("--start", type=int, default=0, help="The first index to start data collection over available glaciers")
parser.add_argument("--stop", type=int, default=-1, help="The last index to finalize data collection over available glaciers")


args = parser.parse_args()

freq          = args.freq
n_trees       = args.n_trees
subset_jjas   = args.subset_jjas
only_t2m_mean = args.only_t2m_mean
start         = args.start
stop          = args.stop

###################

if freq == "M":
    freq_prefix = "monthly"
elif freq == "W":
    freq_prefix = "weekly"

if subset_jjas:
    subset_prefix = "JJAS"
else:
    subset_prefix = "full"

# Dir for results
results_path = f"../results/{freq_prefix}_{subset_prefix}_domain/"
results_path_models = f"../results/{freq_prefix}_{subset_prefix}_domain/trained_models/"
results_path_simulations = f"../results/{freq_prefix}_{subset_prefix}_domain/simulations/"

if not os.path.exists(results_path):
    os.mkdir(results_path)
if not os.path.exists(results_path_models):
    os.mkdir(results_path_models)
if not os.path.exists(results_path_simulations):
    os.mkdir(results_path_simulations)


# Defining sources of data

# RGI IDs
glacier_ids = np.load("../data/misc/glacier_IDs.npy", allow_pickle=True)

# TSLs
tsl_store = pd.read_pickle("../data/tsl/TSLs.pkl")

# Meteo forcing
meteo_path = "../data/meteo/agg-weekly/"

# Physiographic attributes
static_features = pd.read_csv("../data/misc/rgi60_Asia.csv")


# HELPERS

def read_tsl(rgi_id, store=tsl_store):
    
    tsl = tsl_store[tsl_store['RGI_ID'] == rgi_id].copy()
    
    tsl.index = pd.to_datetime(tsl['LS_DATE'])
    
    tsl = pd.DataFrame(tsl['TSL_m'])    
    
    return tsl


def read_meteo(rgi_id, path=meteo_path):
    
    meteo = pd.read_hdf(f"{meteo_path}{rgi_id}.h5")
    meteo.index = pd.to_datetime(meteo['date'])
    meteo = meteo.drop(['date', 'wind_dir_mean_labels'], axis=1)
    
    return meteo


def create_features(dataframe, back_to=12):
    
    # convert circular wind_dir_mean 
    # to two components of cos() and sin()
    # source: https://stats.stackexchange.com/questions/336613/
    # regression-using-circular-variable-hour-from-023-as-predictor
    
    # copy for safety
    df = dataframe.copy()
    
    # create cos() and sin() components
    df["wind_dir_mean_cos"] = np.cos(np.deg2rad(df["wind_dir_mean"]))
    df["wind_dir_mean_sin"] = np.sin(np.deg2rad(df["wind_dir_mean"]))
    
    # drop "wind_dir_mean"
    df = df.drop(["wind_dir_mean"], axis=1)
    
    # make shifts and rolling means
    cols = df.columns
    for col in cols:
        for shift in range(1, back_to+1, 1):
            df["{}-{}".format(col, shift)] = df[col].shift(shift).values
        for rol in range(1, back_to+1, 1):
            df["{}rol-{}".format(col, rol)] = df[col].rolling(window=rol).mean().values
    
    # delete NaNs
    df = df.dropna()
       
    return df


def datasets_construction(rgi_id, freq, subset_jjas, only_t2m_mean):
    
    # get raw TSL measurements
    tsl = read_tsl(rgi_id)
    
    # resample to specific frequency
    tsl_resample = tsl.resample(freq).mean()
    
    # get raw ERA5-Land forcing
    meteo = read_meteo(rgi_id)
    
    # resample to specific frequency
    meteo_resample = pd.DataFrame({'t2m_min'  : meteo['t2m_min'].resample(freq).min(), 
                                   't2m_max'  : meteo['t2m_max'].resample(freq).max(), 
                                   't2m_mean' : meteo['t2m_mean'].resample(freq).mean(), 
                                   'd2m'      : meteo['d2m'].resample(freq).mean(),
                                   
                                   'sp'       : meteo['sp'].resample(freq).mean(),
                                   
                                   'tp'       : meteo['tp'].resample(freq).sum(),
                                   'sf'       : meteo['sf'].resample(freq).sum(),
                                   
                                   'ssrd_mean': meteo['ssrd_mean'].resample(freq).sum(), 
                                   'strd_mean': meteo['strd_mean'].resample(freq).sum(),
                                   
                                    
                                   'wind_mean': meteo['wind_mean'].resample(freq).mean(), 
                                   'wind_dir_mean': meteo['wind_dir_mean'].resample(freq).mean(),
                                   })
    
    if only_t2m_mean:
        meteo_resample = meteo_resample.drop(['t2m_min', 't2m_max'], axis=1)
    
    core_meteo_features = meteo_resample.columns.tolist()
    
    # enrich meteo features
    if freq == "M":
        meteo_enrich = create_features(meteo_resample, back_to=12)
    elif freq == "W":
        meteo_enrich = create_features(meteo_resample, back_to=48) #12 months back considering 4 weeks in each month
    
    
    # meteo for the entire period
    # for model evaluation
    meteo_full = meteo_enrich.dropna()
    
    # merge datasets
    dataset = pd.concat([tsl_resample, meteo_enrich], axis=1)
    
    # drop NaNs
    dataset = dataset.dropna()
    
    if subset_jjas:
        dataset = dataset[(dataset.index.month == 6) | (dataset.index.month == 7) | 
                          (dataset.index.month == 8) | (dataset.index.month == 9)]
    
    return dataset, meteo_full, core_meteo_features


def basin_wise(rgi_id, freq, subset_jjas, only_t2m_mean):
    
    data, _, core_features = datasets_construction(rgi_id, freq, subset_jjas, only_t2m_mean)
        
    static_features_slice = static_features[static_features["RGIId"]==rgi_id].copy()
    
    static_features_slice = static_features_slice[['CenLon', 'CenLat', 'Area', 'Zmin', 'Zmax', 'Zmed', 
                                                   'Slope', 'Aspect', 'Lmax']].copy()
    
    for c in static_features_slice.columns:
        data[c] = static_features_slice[c].values[0] 
        
    core_features = core_features + static_features_slice.columns.tolist()
    
    return data, core_features


def tiny_prep(df):
       
    df_X = df.drop(["TSL_m"], axis=1)
    df_y = df["TSL_m"]
    
    return df_X, df_y


def calculate_importances(imp_instance, core_features):
    
    cols = imp_instance.columns.tolist()
    
    core_feature_cols = {key: [i for i in cols if key in i] for key in core_features}
    
    """
    # temperature-based
    t2m_min_cols = [i for i in cols if "t2m_min" in i]
    t2m_max_cols = [i for i in cols if "t2m_max" in i]
    t2m_mean_cols = [i for i in cols if "t2m_mean" in i]
    
    # precipitation-based
    tp_cols = [i for i in cols if "tp" in i]
    sf_cols = [i for i in cols if "sf" in i]
    
    # surface solar radiation downwards
    ssrd_cols = [i for i in cols if "ssrd" in i]
    
    # surface thermal radiation downwards
    strd_cols = [i for i in cols if "strd" in i]
    
    # wind-based
    wind_max_cols = [i for i in cols if "wind_max" in i]
    wind_mean_cols = [i for i in cols if "wind_mean" in i]
    wind_dir_mean_cols = [i for i in cols if "wind_dir_mean" in i]
    
    # total cloud cover
    tcc_cols = [i for i in cols if "tcc" in i]
    """
    var_importances = []
    
    """
    for var in [t2m_min_cols, 
                t2m_max_cols, 
                t2m_mean_cols, 
                tp_cols, 
                sf_cols,
                ssrd_cols, 
                strd_cols, 
                wind_max_cols, 
                wind_mean_cols, 
                wind_dir_mean_cols,
                tcc_cols]:
    """    
    for var in core_feature_cols.values():
        
        var_importances.append(imp_instance[var].sum(axis=0).sum())
        
    var_importances = np.array(var_importances)
    
    var_importances = var_importances / var_importances.sum()
    
    return var_importances


# calculate the core feature set for further utilization
_, core_features = basin_wise("RGI60-13.00014", freq, subset_jjas, only_t2m_mean)

# begin the countdown
time_start = time.time()

print("Data collections in progress...")

data_file = os.path.join(results_path, "data.csv")

if os.path.isfile(data_file):
    os.remove(data_file)

for i, idx in enumerate(glacier_ids[start:stop]):

    chunk, _ = basin_wise(idx, freq, subset_jjas, only_t2m_mean)
    
    if i == 0:
        chunk.to_csv(data_file, mode="a", index=False, header=True)
    else:
        chunk.to_csv(data_file, mode="a", index=False, header=False)

print("Data preparation has been finished.")


# read the final data (could be huge)
data = pd.read_csv(data_file)


# Data preparation
X_df, y_df = tiny_prep(data)
X, y = X_df.to_numpy(), y_df.to_numpy()


print("Modeling in progress...")

# split-sample tests (cross-validation)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

importances = []
train_scores = []
test_scores = []

for i, (train_index, test_index) in enumerate(kf.split(X)):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # fit model
    m = RandomForestRegressor(n_jobs=-1, n_estimators=n_trees, random_state=42).fit(X_train, y_train)
    
    # save model
    pickle.dump(m, open(os.path.join(results_path_models, f"RF_{i}.pkl") , "wb"))
    
    # make predictions for the test set
    y_test_pred = m.predict(X_test)
    
    # save predictions for the test test
    sim_df = pd.DataFrame({"TSL_obs": y_test, "TSL_sim": y_test_pred})
    sim_df.to_csv(os.path.join(results_path_simulations, f"RF_{i}.csv"))    
    
    # compute scores
    train_score = m.score(X_train, y_train)
    test_score = m.score(X_test, y_test)
    
    # add scores to holders
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"Fold {i+1}: train {np.round(train_score, 2)}, test {np.round(test_score, 2)}")
    
    # calculate feature imortances
    fi = m.feature_importances_
    
    # convert importances to dataframe
    fi_df = pd.DataFrame({0: fi}, index=X_df.columns).T

    # collect
    importances.append(fi_df)


# calculate feature importances that have been averaged over the splits
FI = calculate_importances(pd.concat(importances, axis=0, ignore_index=True), core_features)

# postprocess and save
FI = pd.DataFrame(FI.reshape(1,-1), columns=core_features, index=['domain'])
FI.to_csv(os.path.join(results_path, "importances.csv"))


# save scores
scores = pd.DataFrame({"Train": train_scores, "Test": test_scores})
scores.to_csv(os.path.join(results_path, "scores.csv"))

time_end = time.time()

print(f"Modeling is finished for {np.round( (time_end-time_start)/60 , 1)} minutes")