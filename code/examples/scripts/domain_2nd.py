import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

#key = "monthly_full"
key = "weekly_JJAS"

data = pd.read_csv(f"../data/domain/{key}.csv", index_col=0, header=None)

X = data.drop([310], axis=1).values
y = data[310].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    m = RandomForestRegressor(n_jobs=-1).fit(X_train, y_train)

    pickle.dump(m, open(f"../results/tsl_ML/domain/RF_{key}_{i}.pkl", "wb"))

    print("R2 train: ", m.score(X_train, y_train))
    print("R2 test: ", m.score(X_test, y_test))

    fi = m.feature_importances_
    np.save(f"../results/tsl_ML/domain/RF_fi_{key}_{i}.npy", fi)

