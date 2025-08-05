import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.ml.utils import *
from src.ml.lightgbm_model import train, predict


if __name__ == "__main__":

    # Load Data
    SCRIPT_DIR = Path(__file__).parent
    data_path = SCRIPT_DIR.parent.parent.parent / "data"
    data_file = data_path / "from_bernadette/processed_data.csv"
    result_path = SCRIPT_DIR / "results"
    result_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)

    X, y = get_data(df)

    selected_features = [15, 2, 5, 3, 6, 1, 4, 0]
    X = X[:, selected_features]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # LightGBM hyperparameters for tuning
    param_grid = {
        'num_leaves': [15, 31, 51],  # Smaller num_leaves → simpler trees
        'min_data_in_leaf': [10, 20, 50],  # Larger → forces leaves to have more data (regularization)
        'feature_fraction': [0.7, 0.8, 0.9],  # Subsampling features prevents overfitting
        'bagging_fraction': [0.7, 0.8, 0.9],  # Subsampling data per tree
        'bagging_freq': [1, 5],  # Frequency of bagging
        'lambda_l1': [0.1, 0.5, 1.0],  # L1 Regularization → sparsity
        'lambda_l2': [0.1, 0.5, 1.0, 5.0],  # L2 Regularization → smoothness
        'learning_rate': [0.001, 0.005, 0.01],  # Smaller learning rate → better generalization
        'n_estimators': [500, 1_000, 2_000, 3_000],  # More trees to compensate for smaller learning rate
    }

    # Train Lower Quantile (5%)
    print("\n--- Tuning Lower Quantile Model ---")
    model_lower, params_lower = train(X_train, y_train, X_test, y_test, param_grid, alpha=0.05)

    # Train Median (50%)
    print("\n--- Tuning Median Model ---")
    params_median = params_lower.copy()
    params_median['objective'] = 'regression'
    params_median.pop('alpha')
    model_median = lgb.LGBMRegressor(**params_median)
    model_median.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(100)])
    # model_median, params_median = random_search_lgb(X_train, y_train, X_test, y_test, param_grid, alpha=0.5)

    # Train Upper Quantile (95%)
    print("\n--- Tuning Upper Quantile Model ---")
    params_upper = params_lower.copy()
    params_upper['alpha'] = 0.95
    model_upper = lgb.LGBMRegressor(**params_upper)
    model_upper.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(100)])
    # model_upper, params_upper = random_search_lgb(X_train, y_train, X_test, y_test, param_grid, alpha=0.95)

    # Save Models
    models = {
        "median": model_median,
        "upper": model_upper,
        "lower": model_lower,
    }
    for (key, model) in models.items():
        model.booster_.save_model(result_path / f"lgb_model_{key}.txt")

    predictions = {key: predict(model, X_train, y_train, X_test, y_test) for (key, model) in models.items()}

    # Plot Predictions
    X_lr_train = X_train[:, [5, 1]]
    X_lr_test = X_test[:, [5, 1]]
    lr_predictions = lr_model(X_lr_train, y_train, X_lr_test, y_test)
    plot_predictions(predictions["median"], lr_predictions, result_path, "LightGBM Quantile Regression")


    # testing_flag = predictions["testing_flag"]
    #
    # y = predictions["y"]
    # y_train = y[~testing_flag]
    # y_test = y[testing_flag]
    #
    # x_points = np.arange(1, testing_flag.size+1)
    # x_points_train = x_points[~testing_flag]
    # x_points_test = x_points[testing_flag]
    #
    # y_pred_median = predictions["median"]
    # y_pred_lower = predictions["lower"]
    # y_pred_upper = predictions["upper"]
    #
    # y_err_lower = np.abs(y_pred_lower - y_pred_median)
    # y_err_upper = np.abs(y_pred_upper - y_pred_median)
    #
    # # Make plots
    # fig = plt.figure(figsize=(10, 6))
    # plt.scatter(x_points_train, y_train, color="b", marker="x", label="Training")
    # plt.errorbar(
    #     x_points_train,
    #     y_pred_median[~testing_flag],
    #     yerr=[y_err_lower[~testing_flag], y_err_upper[~testing_flag]],
    #     fmt='o', ecolor='b', alpha=0.7, capsize=3, label="Prediction with 90% Interval"
    # )
    #
    # plt.scatter(x_points_test, y_test, color="r", marker="x", label="Testing")
    # plt.errorbar(
    #     x_points_test,
    #     y_pred_median[testing_flag],
    #     yerr=[y_err_lower[testing_flag], y_err_upper[testing_flag]],
    #     fmt='o', ecolor='r', alpha=0.7, capsize=3, label="Prediction with 90% Interval"
    # )
    #
    # plt.xlabel("Point ID")
    # plt.ylabel("Predicted Values")
    # plt.legend()
    # plt.title("LightGBM Quantile Regression Prediction Intervals")
    # fig.savefig(result_path/"lightgbm_quantile_regression.png")
    #
    #
    # fig = plt.figure(figsize=(8, 6))
    # plt.scatter(y_train, y_pred_median[~testing_flag], c="b", label="Train")
    # plt.scatter(y_test, y_pred_median[testing_flag], c="r", label="Test")
    # plt.axline([0, 0], slope=1, c="k", linestyle="--")
    # plt.xlabel("Observations", fontsize=12)
    # plt.ylabel("Predictions", fontsize=12)
    # plt.legend()
    # plt.suptitle(
    #     f"Training $R^2$ = {predictions['r2_train']:.2f}\n" +
    #     f"Testing $R^2$ = {predictions['r2_test']:.2f}\n" +
    #     f"Entire dataset $R^2$ = {predictions['r2_all']:.2f}"
    # )
    # fig.savefig(result_path/"lightgbm_quantile_predictions.png")
    #
    #
    # # importance = model.feature_importances_
    # # sorted_idx = np.argsort(importance)[::-1]
    # # top_k = 8  # You can try 5, 6, or tune this number
    # # selected_features_idx = sorted_idx[:top_k]
    # # feature_names = df.columns[selected_features]
    #
    #
    # void_ratio_bin = pd.qcut(df["HR"], q=10, labels=False, duplicates='drop')
    #
    # figs = []
    # for void_ratio in np.unique(void_ratio_bin):
    #
    #     df_plot = df.copy()
    #     df_plot = df.loc[void_ratio_bin==void_ratio]
    #     df_plot = df_plot.sort_values(by="leeftijd")
    #     df_plot["Bitumen-gehalte NEN"] = pd.to_numeric(df["Bitumen-gehalte NEN"], errors="coerce")
    #     df_plot["HR"] = df_plot["HR"].mean()
    #     df_plot["Bitumen-gehalte NEN"] = df["Bitumen-gehalte NEN"].mean()
    #
    #     X_plot, _ = prepare_data(df_plot)
    #     X_plot = X_plot[:, selected_features]
    #
    #     min_void_ratio = df_plot["HR"].min()
    #     maX_plot = df_plot["HR"].max()
    #
    #     y_pred_median = model_median.predict(X_plot)
    #     y_pred_upper = model_upper.predict(X_plot)
    #     y_pred_lower = model_lower.predict(X_plot)
    #
    #     fig = plt.figure()
    #     plt.scatter(df_plot["leeftijd"], df_plot["Buigtreksterkte"], color="r", marker="x")
    #     plt.plot(X_plot[:, -1], y_pred_median, c="b")
    #     plt.plot(X_plot[:, -1], y_pred_lower, c="b", linewidth=.5)
    #     plt.plot(X_plot[:, -1], y_pred_upper, c="b", linewidth=.5)
    #     plt.fill_between(X_plot[:, -1], y_pred_lower, y_pred_upper, color="b", alpha=0.3)
    #     plt.xlabel("Age [yr]", fontsize=12)
    #     plt.ylabel("Strength [kPa]", fontsize=12)
    #     plt.grid()
    #     fig.suptitle(f"Void ratio: {min_void_ratio:.1f}-{max_void_ratio:.1f}", fontsize=14)
    #     figs.append(fig)
    #
    # pp = PdfPages(result_path/"lightgbm_quantile_ageplot.pdf")
    # [pp.savefig(fig) for fig in figs]
    # pp.close()

