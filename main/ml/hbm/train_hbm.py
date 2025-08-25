import pymc as pm
import numpy as np
import pandas as pd
import pytensor.tensor as pt
from pathlib import Path
from src.ml.hbm import HBModel


def predict(idata, x, type="partial"):

    if type == "partial":
        y_hat_mean, y_hat_quantiles = predict_partially_pooled(idata, x)
    elif type == "pooled":
        y_hat_mean, y_hat_quantiles = predict_pooled(idata, x)
    elif type == "unpooled":
        y_hat_mean, y_hat_quantiles = predict_unpooled(idata, x)

    return y_hat_mean, y_hat_quantiles


def predict_partially_pooled(idata, x):

    intercept = idata["posterior"].intercept.values
    n_groups = intercept.shape[-1]
    intercept = intercept.reshape(-1, n_groups)
    slope = idata["posterior"].slope.values.reshape(-1, n_groups, x.shape[-1])

    y_hat = intercept + np.sum(slope[np.newaxis, ...]*x[:, np.newaxis, np.newaxis], axis=-1)

    y_hat = y_hat.transpose(1, 2, 0)
    y_hat_mean = y_hat.mean(axis=0)
    y_hat_quantiles = np.quantile(y_hat, [0.05, 0.95], axis=0)

    return y_hat_mean, y_hat_quantiles


if __name__ == "__main__":

    SCRIPT_PATH = Path(__file__).parent
    BRONZE_DATA_PATH = SCRIPT_PATH.parent.parent.parent / "data/from_bernadette/bronze"
    RESULT_PATH = SCRIPT_PATH.parent.parent.parent / "results/ml/hbm"
    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(BRONZE_DATA_PATH/"Database WAB - overzicht ADL28062023_BWich_selection18.8.xlsx")

    columns = {
        "Dijknaam": "dike",
        "Projectnummer": "project",
        "leeftijd": "age",
        "HR": "void_ratio",
        "Bitumen-gehalte NEN": "bitumen",
        "Buigtreksterkte": "strength"
    }
    df = df[list(columns.keys())]
    df = df.rename(columns=columns)
    df["dike_project"] = df["dike"].astype(str) + "-" + df["project"].astype(str)
    df = df[["dike", "project", "dike_project", "age", "void_ratio", "bitumen", "strength"]]
    df = df.dropna(subset="age")
    _, dike_idxs, dike_counts = np.unique(df["dike"], return_inverse=True, return_counts=True)
    df["group"] = dike_idxs
    df["unique_counts"] = dike_counts[dike_idxs]
    df["group"] = np.where(df["unique_counts"] > 5, df["group"], np.nan)
    df = df.dropna(how="any")
    df = df.reset_index(drop=True)
    for i, unique_group in enumerate(sorted(pd.unique(df["group"]))):
        df.loc[df["group"] == unique_group, "group"] = i
    df["group"] = df["group"].astype(int)

    X = df[["age", "void_ratio"]].values
    group_idx = df["group"].values
    n_groups = group_idx.max() + 1
    y = df["strength"].values

    mesh = np.meshgrid(
        np.linspace(df["age"].min(), df["age"].max(), 10),
        np.linspace(df["void_ratio"].min(), df["void_ratio"].max(), 10)
    )
    X_pred = np.c_[[m.flatten() for m in mesh]].T

    model = HBModel("partially_pooled")
    model.train(X, y, group_idx, RESULT_PATH)
    model.eval()

