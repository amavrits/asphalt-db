import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.ml.utils import *
from src.ml.mlp_model import *
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Load Data
    SCRIPT_DIR = Path(__file__).parent
    data_path = SCRIPT_DIR.parent.parent.parent / "data"
    data_file = data_path / "from_bernadette/processed_data.csv"
    result_path = SCRIPT_DIR / "results"
    result_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)

    X, y = get_data(df)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train model
    epochs = 200_000
    lr = 1e-4
    hidden_layers = [256, 128, 64, 32]
    model = MLPRegressor(input_dim=X.shape[1], hidden_layers=hidden_layers, dropout_rate=0.2)
    model.fit(X_train, y_train, epochs, lr)
    torch.save(model.state_dict(), result_path / "model.pth")

    # Evaluate
    predictions = predict(model, X_train, y_train, X_test, y_test)

    # Plot Predictions
    X_lr_train = X_train[:, [1, 2]]
    X_lr_test = X_test[:, [1, 2]]
    lr_predictions = lr_model(X_lr_train, y_train, X_lr_test, y_test)
    plot_predictions(predictions, lr_predictions, result_path, "MLP Regression")


