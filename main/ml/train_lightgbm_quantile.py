import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from src.ml.utils import *


if __name__ == "__main__":

    # Load Data
    SCRIPT_DIR = Path(__file__).parent
    data_path = SCRIPT_DIR.parent.parent / "data"
    data_file = data_path / "from_bernadette/Database WAB - overzicht ADL28062023_BWich_selection18.8.xlsx"
    result_path = Path("results")
    result_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(data_file)

    X, y = prepare_data(df)

    selected_features = [0, 1, 2, 3, 4, 5, 6, 15]  # Pruned Feature Indices
    X = X[:, selected_features]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # LightGBM Parameters (Base Config)
    base_params = {
        'learning_rate': 0.01,
        'num_leaves': 31,
        'min_data_in_leaf': 5,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1
    }

    # Lower Quantile Model (5%)
    params_lower = base_params.copy()
    params_lower['objective'] = 'quantile'
    params_lower['alpha'] = 0.05
    model_lower = lgb.LGBMRegressor(**params_lower, n_estimators=1000)
    model_lower.fit(X_train, y_train)

    # Median Model (Standard Regression)
    params_median = base_params.copy()
    params_median['objective'] = 'regression'
    model_median = lgb.LGBMRegressor(**params_median, n_estimators=1000)
    model_median.fit(X_train, y_train)

    # Upper Quantile Model (95%)
    params_upper = base_params.copy()
    params_upper['objective'] = 'quantile'
    params_upper['alpha'] = 0.95
    model_upper = lgb.LGBMRegressor(**params_upper, n_estimators=1000)
    model_upper.fit(X_train, y_train)

    # Predictions
    y_hat_lower = model_lower.predict(X_train)
    y_hat_median = model_median.predict(X_train)
    y_hat_upper = model_upper.predict(X_train)
    y_pred_lower = model_lower.predict(X_test)
    y_pred_median = model_median.predict(X_test)
    y_pred_upper = model_upper.predict(X_test)
    y_all_lower = model_lower.predict(X)
    y_all_median = model_median.predict(X)
    y_all_upper = model_upper.predict(X)

    # Interval Coverage Evaluation
    coverage_train = np.mean((y_train >= y_hat_lower) & (y_train <= y_hat_upper))
    coverage_test = np.mean((y_test >= y_pred_lower) & (y_test <= y_pred_upper))
    coverage_all = np.mean((y >= y_all_lower) & (y <= y_all_upper))
    r2_train = r2_score(y_train, y_hat_median)
    r2_test = r2_score(y_test, y_pred_median)
    r2_all = r2_score(y, y_all_median)

    print(f"Training Coverage: {coverage_train * 100:.2f}%")
    print(f"Testing Coverage: {coverage_test * 100:.2f}%")
    print(f"Entire Dataset Coverage: {coverage_all * 100:.2f}%")
    print("Training R^2:" + f"{r2_train:.2f}")
    print("Testing R^2:" + f"{r2_test:.2f}")
    print("Entire Dataset R^2:" + f"{r2_all:.2f}")

    # Save Models
    model_lower.booster_.save_model(result_path / "lgb_model_lower.txt")
    model_median.booster_.save_model(result_path / "lgb_model_median.txt")
    model_upper.booster_.save_model(result_path / "lgb_model_upper.txt")

    # Plot Error Bars
    fig = plt.figure(figsize=(10, 6))

    x_points_test = list(range(len(y_test)))
    y_err_lower = np.abs(y_pred_lower - y_pred_median)
    y_err_upper = np.abs(y_pred_upper - y_pred_median)
    sorted_idx = np.argsort(y_test)
    y_test_sorted = y_test[sorted_idx]
    y_pred_sorted = y_pred_median[sorted_idx]
    y_err_lower_sorted = y_err_lower[sorted_idx]
    y_err_upper_sorted = y_err_upper[sorted_idx]
    plt.scatter(x_points_test, y_test_sorted, color="r", marker="x", label="Observations")
    plt.errorbar(x_points_test, y_pred_sorted,
                 yerr=[y_err_lower_sorted, y_err_upper_sorted],
                 fmt='o', ecolor='r', alpha=0.7, capsize=3, label="Prediction with 90% Interval")

    x_points_train = list(range(len(y_train)))
    y_err_lower = np.abs(y_hat_lower - y_hat_median)
    y_err_upper = np.abs(y_hat_upper - y_hat_median)
    sorted_idx = np.argsort(y_train)
    y_train_sorted = y_train[sorted_idx]
    y_hat_sorted = y_hat_median[sorted_idx]
    y_err_lower_sorted = y_err_lower[sorted_idx]
    y_err_upper_sorted = y_err_upper[sorted_idx]
    plt.scatter(x_points_train, y_train_sorted, color="b", marker="x", label="Observations")
    plt.errorbar(x_points_train, y_hat_sorted,
                 yerr=[y_err_lower_sorted, y_err_upper_sorted],
                 fmt='o', ecolor='b', alpha=0.7, capsize=3, label="Prediction with 90% Interval")

    plt.xlabel("Point ID")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.title("LightGBM Quantile Regression Prediction Intervals")
    fig.savefig(result_path/"lightgbm_quantile_regression.png")

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(y_train, y_hat_median, c="b", label="Train")
    plt.scatter(y_test, y_pred_median, c="r", label="Test")
    plt.axline([0, 0], slope=1, c="k", linestyle="--")
    plt.xlabel("Observations", fontsize=12)
    plt.ylabel("Predictions", fontsize=12)
    plt.legend()
    plt.suptitle(
        f"Training $R^2$ = {r2_train:.2f}\n" +
        f"Testing $R^2$ = {r2_test:.2f}\n" +
        f"Entire dataset $R^2$ = {r2_all:.2f}"
    )
    fig.savefig(result_path/"lightgbm_quantile_predictions.png")

