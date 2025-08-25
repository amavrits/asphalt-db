import pymc as pm
import numpy as np
import pandas as pd
import pytensor.tensor as pt
from pathlib import Path


if __name__ == "__main__":

    SCRIPT_PATH = Path(__file__).parent
    BRONZE_DATA_PATH = SCRIPT_PATH.parent.parent.parent / "data/from_bernadette/bronze"

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
    df["group"] = df["group"].astype(int)
    df["unique_counts"] = dike_counts[dike_idxs]
    df["group"] = np.where(df["unique_counts"] > 5, df["group"], np.nan)
    df = df.dropna(how="any")

    X = df[["age", "void_ratio"]].values
    group_idx = df["group"].values.astype(int)
    n_groups = group_idx.max() + 1
    y_obs = df["strength"].values

    with pm.Model() as model:

        intercept = pm.Normal("intercept", mu=0, sigma=10, shape=(n_groups,))
        slope = pm.Normal("slope", mu=0, sigma=10, shape=(n_groups, 2))

        y_hat = pm.Normal("y_hat", intercept[group_idx] + pt.dot(slope[group_idx], X))

        sigma = pm.HalfNormal("sigma", sigma=10)

        # y = pm.Normal("obs", mu=y_hat, sigma=sigma, observed=y_obs)

    idata = pm.sample(model=model, chains=4, tune=1_000, draws=1_000, cores=24, progressbar=True)







