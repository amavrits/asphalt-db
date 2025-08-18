import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.ml.utils import *
from src.ml.pmlp_model import *
import json
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def train(X_train, y_train, epochs, lr, hidden_layers, path=None, device="cpu", verbose=False):
    model = ProbabilisticMLPRegressor(input_dim=X_train.shape[1], hidden_layers=hidden_layers, dropout_rate=0.2, device=device)
    model.fit(X_train, y_train, epochs, lr, verbose)
    if path is not None:
        torch.save(model.state_dict(), path / "model.pth")
    return model


def make_predictions(model, X_train, y_train, X_test, y_test, log_y=False):
    predictions = {
        "mean": predict(model, X_train, y_train, X_test, y_test, log_y, alpha=0.5),
        "lower": predict(model, X_train, y_train, X_test, y_test, log_y, alpha=0.05),
        "upper": predict(model, X_train, y_train, X_test, y_test, log_y, alpha=0.95)
    }
    return predictions


def make_lr_predictions(X_train, y_train, X_test, y_test, age_column_idx=0, void_ratio_column_idx=1):
    X_lr_train = X_train[:, [age_column_idx, void_ratio_column_idx]]
    X_lr_test = X_test[:, [age_column_idx, void_ratio_column_idx]]
    lr_predictions = lr_model(X_lr_train, y_train, X_lr_test, y_test)
    return lr_predictions


def plot(predictions, lr_predictions, path):
    plot_predictions(predictions["mean"], lr_predictions, path, "Probabilistic MLP Regression")
    plot_quantiles(predictions, path)


def load_data(path):
    X_train = np.load(path/"X_train.npy", allow_pickle=True)
    y_train = np.load(path/"y_train.npy", allow_pickle=True)
    X_test = np.load(path/"X_test.npy", allow_pickle=True)
    y_test = np.load(path/"y_test.npy", allow_pickle=True)
    X_timeline = np.load(path/"X_timeline.npy", allow_pickle=True)
    return X_train, y_train, X_test, y_test, X_timeline


def export_predictions(predictions, file):
    if isinstance(list(predictions.values())[0], dict):
        predictions_ser = {
            key1: {
                key2: val2.tolist() if isinstance(val2, np.ndarray) else val2 for (key2, val2) in val1.items()
            } for (key1, val1) in predictions.items()}
    else:
        predictions_ser = {
            key: val.tolist() if isinstance(val, np.ndarray) else val for (key, val) in predictions.items()
        }
    with open(file, "w") as f:
        json.dump(predictions_ser, f)


def run_pipeline(data_path, result_path, params, log_y=False, verbose=False):

    X_train, y_train, X_test, y_test, X_timeline = load_data(data_path)

    epochs, lr, hidden_layers = params
    if log_y:
        model = train(X_train, np.log(y_train), epochs, lr, hidden_layers, result_path, "cpu", verbose)
    else:
        model = train(X_train, y_train, epochs, lr, hidden_layers, result_path, "cpu", verbose)

    model_predictions = make_predictions(model, X_train, y_train, X_test, y_test, log_y)
    export_predictions(model_predictions, result_path/"model_predictions.json")

    lr_predictions = make_lr_predictions(X_train, y_train, X_test, y_test, age_column_idx=0, void_ratio_column_idx=1)
    export_predictions(lr_predictions, result_path/"lr_predictions.json")

    plot_path = result_path / "plots"
    plot_path.mkdir(exist_ok=True, parents=True)
    plot(model_predictions, lr_predictions, plot_path)

    plot_timelines(model, X_timeline, X_train, y_train, plot_path)

    return model_predictions


if __name__ == "__main__":

    # Load Data
    SCRIPT_DIR = Path(__file__).parent
    data_path = SCRIPT_DIR.parent.parent.parent / "data"
    data_file = data_path / "from_bernadette/processed_data.csv"
    result_path = SCRIPT_DIR.parent.parent.parent / "results/ml/probabilistic_mlp"
    result_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)

    X, y = get_data(df)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train model
    epochs = 2
    lr = 1e-4
    hidden_layers = [256, 128, 64, 32]
    model = train(X_train, y_train, epochs, lr, hidden_layers, result_path, verbose=True)

    # Evaluate
    model_predictions = make_predictions(model, X_train, y_train, X_test, y_test)

    #Plot
    plot_timelines(model, X_timeline, X_train, y_train, result_path)

