import random
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm


def train(X_train, y_train, X_val, y_val, param_grid, alpha, n_iter=20):
    best_model = None
    best_loss = float('inf')
    best_params = None

    for i in tqdm(range(n_iter)):
        params = {key: random.choice(values) for key, values in param_grid.items()}
        params['objective'] = 'quantile'
        params['alpha'] = alpha
        params['verbose'] = -1

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='quantile',
            callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=0)]
        )

        val_pred = model.predict(X_val)
        pinball_loss = np.mean(np.maximum(alpha * (y_val - val_pred), (alpha - 1) * (y_val - val_pred)))

        if pinball_loss < best_loss:
            best_loss = pinball_loss
            best_model = model
            best_params = params

        print(f"Iter {i+1}/{n_iter} — Val Loss: {pinball_loss:.4f} — Best Loss: {best_loss:.4f}")

    return best_model, best_params


def predict(model, X_train, y_train, X_test, y_test):

    X = np.vstack((X_train, X_test))
    y = np.concat((y_train, y_test))

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_all = model.predict(X)

    idx = np.argsort(y)
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

