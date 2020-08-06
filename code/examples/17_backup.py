import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool, cpu_count

ids_tsl = np.load("../data/misc/glacier_IDs.npy", allow_pickle=True).tolist()

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

def dataset_construction(freq, subset_jjas, rgi_id):
    
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
    
    # merge datasets
    dataset = pd.concat([tsl_resample, meteo_enrich], axis=1)
    
    # drop NaNs
    dataset = dataset.dropna()
    
    if subset_jjas:
        dataset = dataset[(dataset.index.month == 6) | (dataset.index.month == 7) | 
                          (dataset.index.month == 8) | (dataset.index.month == 9)]
    
    if freq == "M":
        freq_prefix = "monthly"
    elif freq == "W":
        freq_prefix = "weekly"
    
    if subset_jjas:
        subset_prefix = "JJAS"
    else:
        subset_prefix = "full"
    
    dataset.to_csv(f"../results/data4ml/{freq_prefix}_{subset_prefix}/{rgi_id}.csv", compression="gzip")
    
    #print(rgi_id)
    return rgi_id

def detect_no_tsl_data(rgi_id):
    
    tsl = read_tsl(rgi_id)
    
    try:
        tsl = tsl.resample("M").mean()
    except:
        print(rgi_id)
        return rgi_id

%%time
p = Pool(cpu_count())
no_tsl_data = list(p.imap(detect_no_tsl_data, ids_tsl))

p.close()
p.join()


no_tsl_data = [i for i in no_tsl_data if i is not None]
ids_tsl_valid = [i for i in ids_tsl if i not in no_tsl_data]


# monthly, full data
p = Pool(cpu_count())

freq="M"
subset_jjas = False

func = partial(dataset_construction, freq, subset_jjas)
saved = list(p.imap(func, ids_tsl_valid))

p.close()
p.join()

print(len(saved))