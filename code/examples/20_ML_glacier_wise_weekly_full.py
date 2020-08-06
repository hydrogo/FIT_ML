import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score

import pickle


glacier_ids = np.load("../results/misc/glacier_ids_valid.npy")


tsl_path = "../results/tsl_csv/"
meteo_path = "../data/FIT_forcing/meteo/"


def read_tsl(rgi_id, path=tsl_path):
    
    tsl = pd.read_csv(f"{tsl_path}{rgi_id}.csv", index_col=0, parse_dates=True)
    
    return tsl


def read_meteo(rgi_id, path=meteo_path):
    
    meteo = pd.read_hdf(f"{meteo_path}{rgi_id}.h5")
    
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


def datasets_construction(rgi_id, freq, subset_jjas):
    
    # get raw TSL measurements
    tsl = read_tsl(rgi_id)
    
    # resample to specific frequency
    tsl_resample = tsl.resample(freq).mean()
    
    # get raw ERA5-Land forcing
    meteo = read_meteo(rgi_id)
    
    # resample to specific frequency
    meteo_resample = pd.DataFrame({'t2m_min': meteo['t2m_min'].resample(freq).min(), 
                                   't2m_max': meteo['t2m_max'].resample(freq).max(), 
                                   't2m_mean': meteo['t2m_mean'].resample(freq).mean(), 
                                   'tp': meteo['tp'].resample(freq).sum(), 
                                   'sf': meteo['sf'].resample(freq).sum(),
                                   'ssrd': meteo['ssrd'].resample(freq).sum(), 
                                   'strd': meteo['strd_mean'].resample(freq).sum(),
                                   'wind_max': meteo['wind_max'].resample(freq).max(), 
                                   'wind_mean': meteo['wind_mean'].resample(freq).mean(), 
                                   'wind_dir_mean': meteo['wind_dir_mean'].resample(freq).mean(),
                                   'tcc': meteo['tcc'].resample(freq).mean()})
    
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
    
    return dataset, meteo_full


def tiny_prep(df):
       
    df_X = df.drop(["TSL_normalized"], axis=1)
    df_y = df["TSL_normalized"]
    
    return df_X, df_y


def calculate_importances(imp_instance):
    
    cols = imp_instance.columns.tolist()
    
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
    
    var_importances = []
    
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
        
        var_importances.append(imp_instance[var].sum(axis=0).sum())
        
    var_importances = np.array(var_importances)
    
    var_importances = var_importances / var_importances.sum()
    
    return var_importances


def ML_me(rgi_id, freq, subset_jjas):
    
    # data: features and target
    df, meteo_full = datasets_construction(rgi_id, freq, subset_jjas)
    
    features_df, target_df = tiny_prep(df)
    features, target = features_df.values, target_df.values 
    
    # model: RandomForest for regression
    # parsimonious model with low complexity
    # n_estimators is better to be multiple of available CPU threads
    model = RandomForestRegressor(random_state=76, n_estimators=8, n_jobs=-1)
    
    # holders for each step predictions/observations
    obs = []
    prd = []
    
    # leave-one-out cross-validation
    loo = LeaveOneOut()
    
    # predictions for whole interval
    full_preds = []
    
    # feature importances holder
    importances = []
    
    # misc prefixes
    if freq == "M":
        freq_prefix = "monthly"
    elif freq == "W":
        freq_prefix = "weekly"
    
    if subset_jjas:
        subset_prefix = "JJAS"
    else:
        subset_prefix = "full"
    
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
                
        # fit eli5 to calculate permutation imporatnces
        #pi = PermutationImportance(estimator=model, cv="prefit", n_iter=1).fit(X_train, y_train)
        
        # get feature importances from eli5
        #fi = pi.feature_importances_
        
        # get feature importances from native model
        fi = model.feature_importances_
        
        # convert importances to dataframe
        fi_df = pd.DataFrame({0: fi}, index=features_df.columns).T
        
        # collect
        importances.append(fi_df)
        
        # save model instance
        #pickle.dump(model, open(f"../results/tsl_ML/{freq_prefix}_{subset_prefix}/trained_models/{rgi_id}_{test}.pkl", 'wb'))
                  
    
    # grab predictions together
    obs = np.array(obs)
    prd = np.array(prd)
    
    # calculate r2 LOO score
    loo_score = r2_score(obs, prd)
    
    # grab loo obs and preds together
    loo_sim = pd.DataFrame({"obs": obs, "sim": prd}, index=features_df.index)
    # save loo predicitions
    loo_sim.to_csv(f"../results/tsl_ML/{freq_prefix}_{subset_prefix}/tsl_simulated/{rgi_id}_loo.csv", 
                   compression="gzip")
    
    # postprocessing of entire predictions
    full_preds = np.array(full_preds).reshape(len(target), -1)
    ensemble_mean = full_preds.mean(axis=0)
    tsl_for_meteo_full = pd.DataFrame({"TSL_sim": ensemble_mean}, index=meteo_full.index)
    # save ensemble mean for the entire meteo ts
    tsl_for_meteo_full.to_csv(f"../results/tsl_ML/{freq_prefix}_{subset_prefix}/tsl_simulated/{rgi_id}_ens.csv", 
                              compression="gzip")
    
    # get importances together
    all_importances = pd.concat(importances, axis=0, ignore_index=True)
    
    # calculate relative importances
    rel_importances = calculate_importances(all_importances)
    
    #return loo_sim, loo_score, rel_importances, tsl_for_meteo_full
    return loo_score, rel_importances


list_loo_scores = []
list_importances = []

broken = []

for num, idx in enumerate(glacier_ids):
        
    try:
        individual_loo_score, individual_importances = ML_me(idx, "W", False)
    
    except:
        print(idx, "Warning")
        broken.append(idx)
        
    list_loo_scores.append(individual_loo_score)
    list_importances.append(individual_importances)
    
    print(num, 
          idx, 
          np.round(individual_loo_score, 2),  
          np.argmax(individual_importances))


scores_loo_arr = np.array(list_loo_scores)
importances_arr = np.array(list_importances)


res = pd.DataFrame({"id": glacier_ids, 
                    "t2m_min":  importances_arr[:, 0], 
                    't2m_max':  importances_arr[:, 1],
                    't2m_mean': importances_arr[:, 2],
                    'tp':       importances_arr[:, 3], 
                    'sf':       importances_arr[:, 4],
                    'ssrd':     importances_arr[:, 5],
                    'strd':     importances_arr[:, 6],
                    'wind_max': importances_arr[:, 7],
                    'wind_mean':importances_arr[:, 8], 
                    "wind_dir_mean": importances_arr[:, 9],
                    'tcc':      importances_arr[:, 10],
                    "score_loo":scores_loo_arr})


print(res.describe())

try:
    res.to_csv("../results/tsl_ML/weekly_full/misc/weekly_full_drivers.csv")
except:
    res.to_csv("weekly_full_drivers.csv")
