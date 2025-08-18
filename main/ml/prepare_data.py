import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def generate_base_feats(df):
    df["age_x_void"] = df["age"] * df["void_ratio"]
    df["age_squared"] = df["age"] ** 2
    df["void_squared"] = df["void_ratio"] ** 2
    df["log_age"] = np.log1p(df["age"])
    df["log_void"] = np.log1p(df["void_ratio"])
    df["inv_age"] = 1 / (df["age"] + 1)
    df["inv_void"] = 1 / (df["void_ratio"] + 1)
    return df


def generate_bitumen_feats(df):
    df_bitumen = df.copy()
    df_bitumen = df_bitumen[pd.to_numeric(df_bitumen["bitumen"], errors="coerce").notna()]
    df_bitumen["bitumen"] = df_bitumen["bitumen"].astype(float)
    df_bitumen["age_x_bitumen"] = df_bitumen["age"] * df["bitumen"]
    df_bitumen["bitumen_squared"] = df_bitumen["bitumen"] ** 2
    df_bitumen["void_x_bitumen"] = df_bitumen["void_ratio"] * df_bitumen["bitumen"]
    df_bitumen["bitumen_per_void"] = df_bitumen["bitumen"] / (df_bitumen["void_ratio"] + 1e-6)
    df_bitumen["log_bitumen"] = np.log1p(df_bitumen["bitumen"])
    df_bitumen["mean_feature"] = df_bitumen[["age", "void_ratio", "bitumen"]].mean(axis=1)
    return df_bitumen


def generate_no_bitumen_feats(df):
    df_nobitumen = df.copy()
    df_nobitumen = df_nobitumen.drop(columns=["bitumen"])
    df_nobitumen["mean_feature"] = df_nobitumen[["age", "void_ratio"]].mean(axis=1)
    return df_nobitumen


def generate_timeline_data(df, feat_cols):

    use_bitumen = any(["bitumen" in feat for feat in feat_cols])

    df_timeline = pd.DataFrame(
        data=np.linspace(df["age"].min(), df["age"].max(), 1_000),
        columns=["age"]
    )

    if use_bitumen:
        df_bitumen = generate_bitumen_feats(df)
        df_timeline["bitumen"] = df_bitumen["bitumen"].mean()
    else:
        df_timeline["bitumen"] = 1

    df_timeline["void_ratio"] = 0
    void_ratio_deciles = pd.qcut(df["void_ratio"], q=10, labels=False).values
    df_timelines = []
    for (void_ratio_decile_bottom, void_ratio_decile_top) in zip(void_ratio_deciles[:-1], void_ratio_deciles[1:]):
        df_timeline_decile = df_timeline.copy()
        df_timeline_decile["void_ratio"] = (void_ratio_decile_bottom + void_ratio_decile_top) / 2
        df_timelines.append(df_timeline_decile)
    df_timeline = pd.concat(df_timelines)

    df_timeline = generate_base_feats(df_timeline)
    if use_bitumen:
        df_timeline = generate_bitumen_feats(df_timeline)
    else:
        df_timeline = generate_no_bitumen_feats(df_timeline)

    X = df_timeline[feat_cols].values

    return X


def generate_splits(df, feat_cols, target_col, path, n_splits=100):

    X = df_bitumen[feat_cols].values
    y = df_bitumen[target_col].values

    X_timeline = generate_timeline_data(df, feat_cols)

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
        np.save(split_path/"X_timeline.npy", X_timeline)


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

    # Generate features
    df = generate_base_feats(df)

    # Save with "Bitumen"
    GOLD_BITUMEN_PATH = GOLD_DATA_PATH / "w_bitumen"
    df_bitumen = generate_bitumen_feats(df)
    df_bitumen.to_csv(SILVER_DATA_PATH/"w_bitumen.csv")
    feat_cols = [col for col in df_bitumen.columns if col not in ["dike", "project", "dike_project", "strength"]]
    target_col = "strength"
    generate_splits(df_bitumen, feat_cols, target_col, GOLD_BITUMEN_PATH, n_splits)

    # Save without "Bitumen"
    GOLD_NOBITUMEN_PATH = GOLD_DATA_PATH / "wo_bitumen"
    df_nobitumen = generate_no_bitumen_feats(df)
    df_nobitumen.to_csv(SILVER_DATA_PATH/"wo_bitumen.csv")
    feat_cols = [col for col in df_nobitumen.columns if col not in ["dike", "project", "dike_project", "strength"]]
    target_col = "strength"
    generate_splits(df_nobitumen, feat_cols, target_col, GOLD_NOBITUMEN_PATH, n_splits)


