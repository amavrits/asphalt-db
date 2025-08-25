import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sympy import false

from src.ml.xgboost_model import *
import json
import joblib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def make_predictions(model, X_train, y_train, X_test, y_test, log_y=False):
    predictions = {
        "mean": predict(model, X_train, y_train, X_test, y_test, log_y),
        "lower": predict(model, X_train, y_train, X_test, y_test, log_y),
        "upper": predict(model, X_train, y_train, X_test, y_test, log_y)
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

    if log_y:
        model = train(X_train, np.log(y_train), params)
    else:
        model = train(X_train, y_train, params)

    joblib.dump(model, result_path/"model.json")

    model_predictions = make_predictions(model, X_train, y_train, X_test, y_test, log_y)
    export_predictions(model_predictions, result_path/"model_predictions.json")

    lr_predictions = make_lr_predictions(X_train, y_train, X_test, y_test, age_column_idx=0, void_ratio_column_idx=1)
    export_predictions(lr_predictions, result_path/"lr_predictions.json")

    plot_path = result_path / "plots"
    plot_path.mkdir(exist_ok=True, parents=True)
    plot(model_predictions, lr_predictions, plot_path)

    plot_model_timelines(model, X_timeline, X_train, y_train, plot_path, quantiles=False)

    return model_predictions


if __name__ == "__main__":

    pass

