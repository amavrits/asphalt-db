import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xgboost as xgb
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

    # Hyperparameter Grid
    param_grid = {
        'n_estimators': [500, 1000, 2000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1.0, 1.5]
    }

    # Randomized Search with Early Stopping
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
    search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_grid,
        n_iter=50,
        scoring="neg_root_mean_squared_error",
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    model = search.best_estimator_

    # Evaluate
    y_hat = model.predict(X_train)
    y_pred = model.predict(X_test)
    y_pred_all = model.predict(X)

    # Plot Predictions
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(y_train, y_hat, c="b", label="Train")
    plt.scatter(y_test, y_pred, c="r", label="Test")
    plt.axline([0, 0], slope=1, c="k", linestyle="--")
    plt.xlabel("Observations", fontsize=12)
    plt.ylabel("Predictions", fontsize=12)
    plt.legend()
    plt.suptitle(
        f"Training $R^2$ = {r2_score(y_train, y_hat):.2f}\n" +
        f"Testing $R^2$ = {r2_score(y_test, y_pred):.2f}\n" +
        f"Entire dataset $R^2$ = {r2_score(y, y_pred_all):.2f}"
    )
    fig.savefig(result_path/"")

    # Feature Importance Plot
    # xgb.plot_importance(model, importance_type='gain', height=0.5, max_num_features=10)
    # plt.title("Top Feature Importances")
    # plt.show()

    # Save the model
    model.save_model(result_path / "xgb_model.json")

