import pandas as pd
import numpy as np


def prepare_data(df):
    columns = {
        "Dijknaam": "dijk",
        "Projectnummer": "project",
        "leeftijd": "age",
        "HR": "void_ratio",
        "Bitumen-gehalte NEN": "bitumen",
        "Buigtreksterkte": "str"
    }
    df = df[list(columns.keys())]
    df = df.rename(columns=columns)
    df["bitumen"] = pd.to_numeric(df["bitumen"], errors="coerce")

    # Multiplied features
    df['age_x_void'] = df['age'] * df['void_ratio']
    df['age_x_bitumen'] = df['age'] * df['bitumen']
    df['void_x_bitumen'] = df['void_ratio'] * df['bitumen']
    df['bitumen_per_void'] = df['bitumen'] / (df['void_ratio'] + 1e-6)  # Avoid division by zero

    # Polynomial Features
    df['age_squared'] = df['age'] ** 2
    df['void_squared'] = df['void_ratio'] ** 2
    df['bitumen_squared'] = df['bitumen'] ** 2

    # Log Features
    df['log_age'] = np.log1p(df['age'])
    df['log_void'] = np.log1p(df['void_ratio'])
    df['log_bitumen'] = np.log1p(df['bitumen'])

    # Reciprocal Features
    df['inv_age'] = 1 / (df['age'] + 1)
    df['inv_void'] = 1 / (df['void_ratio'] + 1)

    # Mean of all features (as aggregate)
    df['mean_feature'] = df[['age', 'void_ratio', 'bitumen']].mean(axis=1)

    df = df.dropna(how="any")

    y = df["str"].values
    df = df.drop(columns=["dijk", "project", "str"])
    X = df.values

    return X, y


if __name__ == "__main__":

    pass

