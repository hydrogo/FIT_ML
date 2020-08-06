import argparse
import os
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score

import pickle


###################
# Global parameters

# arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("--freq", type=str, default="W", help="Temporal resolution: W for weekly, or M for monthly")
parser.add_argument("--n_trees", type=int, default=10, help="Number of decision trees in Random Forest ensemble: 10, 100")
parser.add_argument("--subset_jjas", type=bool, default=False, help="True for subseting only June, July, August, and September. False for considering the entire timeseries")
parser.add_argument("--only_t2m_mean", type=bool, default=True, help="True for considering only mean temperature. False for considering mean, min, max temperatures")
parser.add_argument("--start", type=int, default=0, help="The first index to start the loop over available glaciers")
parser.add_argument("--stop", type=int, default=10, help="The last index to finalize the loop over available glaciers")


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
results_path = f"../results/{freq_prefix}_{subset_prefix}/"
results_path_models = f"../results/{freq_prefix}_{subset_prefix}/trained_models/"
results_path_simulations = f"../results/{freq_prefix}_{subset_prefix}/simulations/"

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
    # you may comment (#) variables that you wish to exclude from the processing
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


def ML_me(rgi_id, freq, subset_jjas, only_t2m_mean):
    
    # data: features and target
    df, meteo_full, core_features = datasets_construction(rgi_id, freq, subset_jjas, only_t2m_mean)
    
    features_df, target_df = tiny_prep(df)
    features, target = features_df.values, target_df.values 
    
    # model: RandomForest for regression
    # parsimonious model with low complexity
    # n_estimators is better to be multiple of available CPU threads
    model = RandomForestRegressor(random_state=76, n_estimators=n_trees, n_jobs=-1) #N_ESTIMATORS
    
    # holders for each step predictions/observations
    obs = []
    prd = []
    
    # leave-one-out cross-validation
    loo = LeaveOneOut()
    
    # predictions for whole interval
    full_preds = []
    
    # feature importances holder
    importances = []
    
        
    # training loop
    for train, test in loo.split(target): 
        
        # split data on train/test
        X_train, y_train, X_test, y_test = features[train], target[train], features[test], target[test]
        
        # fir data on train
        model.fit(X_train, y_train)
        
        # calculate test prediction
        y_pred = model.predict(X_test)
        
        # calculate prediction for the entire dataset
        full_pred = model.predict(meteo_full.values)
        
        # add test and predicted values to holders
        obs.append(y_test[0])
        prd.append(y_pred[0])
        full_preds.append(full_pred)
                
        # get feature importances from native model
        fi = model.feature_importances_
        
        # convert importances to dataframe
        fi_df = pd.DataFrame({0: fi}, index=features_df.columns).T
        
        # collect
        importances.append(fi_df)
        
        # save model instance
        pickle.dump(model, open(os.path.join(results_path_models, f"{rgi_id}_{test}.pkl"), 'wb'))
                  
    
    # grab predictions together
    obs = np.array(obs)
    prd = np.array(prd)
    
    # calculate r2 LOO score
    loo_score = r2_score(obs, prd)
    loo_score = pd.DataFrame(loo_score, columns=['R2'], index=[rgi_id])
    # save r2 LOO score
    loo_score.to_csv(os.path.join(results_path_simulations, f"{rgi_id}_r2.csv"),
                     compression="gzip")
    
    # grab loo obs and preds together
    loo_sim = pd.DataFrame({"obs": obs, "sim": prd}, index=features_df.index)
    # save loo predicitions
    loo_sim.to_csv(os.path.join(results_path_simulations, f"{rgi_id}_loo.csv"), 
                   compression="gzip")
    
    # postprocessing of entire predictions
    full_preds = np.array(full_preds).reshape(len(target), -1)
    ensemble_mean = full_preds.mean(axis=0)
    tsl_for_meteo_full = pd.DataFrame({"TSL_sim": ensemble_mean}, index=meteo_full.index)
    # save ensemble mean for the entire meteo ts
    tsl_for_meteo_full.to_csv(os.path.join(results_path_simulations, f"{rgi_id}_ens.csv"), 
                              compression="gzip")
    
    # get importances together
    all_importances = pd.concat(importances, axis=0, ignore_index=True)
    
    # calculate relative importances
    rel_importances = calculate_importances(all_importances, core_features)
    rel_importances = pd.DataFrame(rel_importances.reshape(1,-1), columns=core_features, index=[rgi_id])
    
    # save relative importances
    rel_importances.to_csv(os.path.join(results_path_simulations, f"{rgi_id}_ri.csv"),
                           compression="gzip")
    
    return loo_sim, loo_score, rel_importances, tsl_for_meteo_full
    #return loo_score, rel_importances


# MAIN ROUTINE

# holder for ids that will not pass the modeling workflow
failed = []

time_start = time.time()
# THE LOOP
for num, idx in enumerate(glacier_ids[start:stop]):
        
    try:
        individual_loo_sim, \
        individual_loo_score, \
        individual_importances, \
        individual_tsl_simulations = ML_me(rgi_id=idx, 
                                           freq=freq, 
                                           subset_jjas=subset_jjas, 
                                           only_t2m_mean=only_t2m_mean)
    
    except:
        print(idx, "Warning: failed to pass the modeling chain")
        failed.append(idx)
          
    print(f"{num+1}/{len(glacier_ids[start:stop])}", 
          idx, 
          np.round(individual_loo_score.to_numpy(), 2),  
          individual_importances.columns[np.argmax(individual_importances.to_numpy())])

time_end = time.time() 

if len(failed) > 0:
    np.save(os.path.join(results_path_simulations, "__failed.npy"), np.array(broken))
    print(f"Number of glaciers failed: {len(failed)}")
    
print(f"Computation is finished for {np.round( (time_end-time_start)/60 , 1)} minutes")
