import numpy as np
import xgboost as xgb
from src.ml.utils import *
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV


def train(X_train, y_train, param_grid, cv=5):

    xgb_model = xgb.XGBRegressor(objective="reg:squarederror")

    search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_grid,
        n_iter=50,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    model = search.best_estimator_

    return model


def predict(model, X_train, y_train, X_test, y_test):

    X = np.vstack((X_train, X_test))
    y = np.concat((y_train, y_test))

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_all = model.predict(X)

    idx = np.argsort(y)[::-1]
    y = y[idx]
    y_pred_all = y_pred_all[idx]

    test_flag = np.zeros(y.size).astype(bool)
    test_flag[-y_test.size:] = 1
    test_flag = test_flag[idx]

    predictions = {
        "y_train": y_train,
        "y_test": y_test,
        "y": y,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_pred_all": y_pred_all,
        "test_flag": test_flag,
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "r2_all": r2_score(y, y_pred_all),
    }

    return predictions




if __name__ == "__main__":

    pass

