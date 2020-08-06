import numpy as np
import pandas as pd
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--key", type=str, help="monthly_full, monthly_JJAS, weekly_full, weekly_JJAS")
args = parser.parse_args()

key = args.key

# for testing
#key = "monthly_full"

freq= key[0].upper() # "M" or "W"


glacier_ids = np.load("../results/misc/glacier_ids_valid.npy")

tsl_path = "../results/tsl_csv/"
meteo_path = "../data/FIT_forcing/meteo/"

static_features = pd.read_csv("../data/misc/rgi60_Asia.csv")


# ML models
# have to load them explicitly before as they are huge

m0 = pickle.load(open(f"../results/tsl_ML/domain/RF_{key}_0.pkl", "rb"))
m1 = pickle.load(open(f"../results/tsl_ML/domain/RF_{key}_1.pkl", "rb"))
m2 = pickle.load(open(f"../results/tsl_ML/domain/RF_{key}_2.pkl", "rb"))
m3 = pickle.load(open(f"../results/tsl_ML/domain/RF_{key}_3.pkl", "rb"))
m4 = pickle.load(open(f"../results/tsl_ML/domain/RF_{key}_4.pkl", "rb"))

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


def dataset_construction(rgi_id, freq):

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
    data = meteo_enrich.dropna()

    static_features_slice = static_features[static_features["RGIId"]==rgi_id].copy()
    
    static_features_slice = static_features_slice[['CenLon', 'CenLat', 'Area', 'Zmin', 'Zmax', 'Zmed', 
                                                   'Slope', 'Aspect', 'Lmax']].copy()
    
    for c in static_features_slice.columns:
        data[c] = static_features_slice[c].values[0]

    return data

# THE LOOP
for i, idx in enumerate(glacier_ids):

    full_meteo = dataset_construction(idx, freq)
    
    predictions = pd.DataFrame({"m0": m0.predict(full_meteo), 
                                "m1": m1.predict(full_meteo), 
                                "m2": m2.predict(full_meteo), 
                                "m3": m3.predict(full_meteo), 
                                "m4": m4.predict(full_meteo)}, index=full_meteo.index)

    predictions.to_csv(f"../results/tsl_ML/domain/tsl_simulated/{key}/{idx}.csv")

    print(i, idx)
