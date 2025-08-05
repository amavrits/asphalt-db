import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import r2_score


def random_search_lgb(X_train, y_train, X_val, y_val, param_grid, alpha, n_iter=20):
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


def lgb_predict(models, X_train, y_train, X_test, y_test):

    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    idx = np.argsort(y)
    testing_flag = np.zeros_like(y)
    testing_flag[X_train.shape[0]:] = 1
    testing_flag = testing_flag[idx]
    testing_flag = testing_flag.astype(bool)
    y = y[idx]

    predictions = {
        "y_train": y_train,
        "y_test": y_test,
        "y": y,
        "testing_flag": testing_flag,
    }

    for (key, model) in models.items():
        y_pred = model.predict(X)
        y_pred = y_pred[idx]
        predictions[key] = y_pred

    predictions["r2_train"] = r2_score(y[~testing_flag], predictions["median"][~testing_flag])
    predictions["r2_test"] = r2_score(y[testing_flag], predictions["median"][testing_flag])
    predictions["r2_all"] = r2_score(y, predictions["median"])

    predictions["coverage_train"] = np.mean((y_train >= predictions["lower"][~testing_flag]) & (y_train <= predictions["upper"][~testing_flag]))
    predictions["coverage_test"] = np.mean((y_test >= predictions["lower"][testing_flag]) & (y_test <= predictions["upper"][testing_flag]))
    predictions["coverage_all"] = np.mean((y >= predictions["lower"]) & (y <= predictions["upper"]))

    return predictions


if __name__ == "__main__":

    pass

