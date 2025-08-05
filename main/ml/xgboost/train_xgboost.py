import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.ml.xgboost_model import train, predict
from src.ml.utils import *
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # Load Data
    SCRIPT_DIR = Path(__file__).parent
    data_path = SCRIPT_DIR.parent.parent.parent / "data"
    data_file = data_path / "from_bernadette/processed_data.csv"
    result_path = SCRIPT_DIR / "results"
    result_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)

    X, y = get_data(df)

    selected_features = [0, 1, 2, 3, 4, 5, 6, 15]
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
    model = train(X_train, y_train, param_grid, cv=5)

    # Save the model
    model.save_model(result_path / "xgb_model.json")

    # Evaluate
    predictions = predict(model, X_train, y_train, X_test, y_test)

    # Plot Predictions
    X_lr_train = X_train[:, [1, 2]]
    X_lr_test = X_test[:, [1, 2]]
    lr_predictions = lr_model(X_lr_train, y_train, X_lr_test, y_test)
    plot_predictions(predictions, lr_predictions, result_path, "XGBoost Regression")


