import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def generate_splits(df, feat_cols, target_col, path, n_splits=100):
    X = df_bitumen[feat_cols].values
    y = df_bitumen[target_col].values
    rng = np.random.default_rng(seed=42)
    for i in range(n_splits):
        split_path = path / f"split_{i+1}"
        split_path.mkdir(parents=True, exist_ok=True)
        seed = rng.integers(0, 9999)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        np.save(split_path/"X_train.npy", X_train)
        np.save(split_path/"X_test.npy", X_test)
        np.save(split_path/"y_train.npy", y_train)
        np.save(split_path/"y_test.npy", y_test)


if __name__ == "__main__":

    n_splits = 100

    SCRIPT_PATH = Path(__file__).parent
    BRONZE_DATA_PATH = SCRIPT_PATH.parent.parent / "data/from_bernadette/bronze"
    SILVER_DATA_PATH = SCRIPT_PATH.parent.parent / "data/from_bernadette/silver"
    SILVER_DATA_PATH.mkdir(exist_ok=True, parents=True)
    GOLD_DATA_PATH = SCRIPT_PATH.parent.parent / "data/from_bernadette/gold"
    GOLD_DATA_PATH.mkdir(exist_ok=True, parents=True)

    bronze_filename = "Database WAB - overzicht ADL28062023_BWich_selection18.8.xlsx"
    df = pd.read_excel(BRONZE_DATA_PATH/bronze_filename)

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

    # Save with "Bitumen"
    df_bitumen = df.copy()
    df_bitumen = df_bitumen[pd.to_numeric(df["bitumen"], errors="coerce").notna()]
    df_bitumen.to_csv(SILVER_DATA_PATH/"w_bitumen.csv")

    GOLD_BITUMEN_PATH = GOLD_DATA_PATH / "w_bitumen"
    feat_cols = ["age", "void_ratio", "bitumen"]
    target_col = "strength"
    generate_splits(df_bitumen, feat_cols, target_col, GOLD_BITUMEN_PATH, n_splits)

    # Save without "Bitumen"
    df_nobitumen = df.copy()
    df_nobitumen = df_nobitumen.drop(columns=["bitumen"])
    df_nobitumen.to_csv(SILVER_DATA_PATH/"wo_bitumen.csv")

    GOLD_NOBITUMEN_PATH = GOLD_DATA_PATH / "wo_bitumen"
    feat_cols = ["age", "void_ratio"]
    target_col = "strength"
    generate_splits(df_nobitumen, feat_cols, target_col, GOLD_NOBITUMEN_PATH, n_splits)


