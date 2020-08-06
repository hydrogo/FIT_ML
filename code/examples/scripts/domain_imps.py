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

meteo_path = "../data/FIT_forcing/meteo/"

static_features = pd.read_csv("../data/misc/rgi60_Asia.csv")


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

    # physiographic characteristics
    # ['CenLon', 'CenLat', 'Area', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Aspect', 'Lmax']

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
                tcc_cols, 
                ['CenLon'], ['CenLat'], ['Area'], ['Zmin'], ['Zmax'], ['Zmed'], ['Slope'], ['Aspect'], ['Lmax']]:

        var_importances.append(imp_instance[var].sum(axis=0).sum())

    var_importances = np.array(var_importances)

    var_importances = var_importances / var_importances.sum()

    df_imps = pd.DataFrame(var_importances.reshape(1, -1), columns=["t2m_min", 't2m_max', 't2m_mean', 'tp', 'sf', 'ssrd', 'strd', 'wind_max', 'wind_mean', "wind_dir_mean", 'tcc', 
                                                     'CenLon', 'CenLat', 'Area', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Aspect', 'Lmax'])

    return df_imps


# retrieve columns
cols = dataset_construction("RGI60-14.24753", freq).columns.tolist()


importances = []

for i in range(5):
   
    imp_from_model = np.load(f"../results/tsl_ML/domain/RF_fi_{key}_{i}.npy")
    importances.append(pd.DataFrame(imp_from_model.reshape(1,-1), columns=cols, index=[0]))

# get importances together
all_importances = pd.concat(importances, axis=0, ignore_index=True)

#print(all_importances, all_importances.shape)

# calculate relative importances
rel_importances = calculate_importances(all_importances)

#print(rel_importances)

rel_importances.to_csv(f"../results/tsl_ML/domain/domain_fi_{key}.csv")
