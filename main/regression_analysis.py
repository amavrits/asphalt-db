import pandas as pd
import numpy as np
from scipy.stats import t
from pathlib import Path
import json
from dataclasses import dataclass, field, asdict
from numpy.typing import NDArray
from typing import List, Dict, Tuple, Optional
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns


def select_dike(df: pd.DataFrame, dike_id: str = "emmapolder") -> str:
    available_dikes = pd.unique(df["dike_name"]).tolist()
    dike_name = [d for d in available_dikes if dike_id in d]
    if len(dike_name) > 1:
        raise ValueError(f"Invalid dike ID, '{dike_id}' is present in multiple dike names {available_dikes}.")
    else:
        dike_name = dike_name[0]
    return dike_name


def filter_data(df: pd.DataFrame, dike_sections: str) -> pd.DataFrame:
    df_filtered = df.copy()
    df_filtered = df_filtered.loc[df_filtered["dike_name"].isin(dike_sections)]
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered["dike_section"] = df_filtered['dike_name'].str.split("_").str[:2].str.join("_").replace(r'[\s\-0-9\,\_]', " ", regex=True).str.strip()
    df_filtered["age"] = df_filtered["investigation_year"] - df_filtered["construction_year"]
    df_filtered["age_diff"] = 0.
    df_filtered["init_sig_b"] = 0.
    df_filtered["sig_b_norm"] = 0.
    for section in pd.unique(df_filtered["dike_section"]):
        df_section = df_filtered.loc[df_filtered["dike_section"] == section]
        ages = df_section["age"].values
        unique_ages = np.unique(df_section["age"])
        age_diffs = ages[:, np.newaxis] - unique_ages
        age_diffs[age_diffs<=0] = 9999
        df_section["age_diff"] = age_diffs.min(axis=-1)
        df_section["age_diff"] = df_section["age_diff"].replace(9999, 0)
        df_filtered.loc[df_filtered["dike_section"] == section, "age_diff"] = age_diffs.min(axis=-1)
        df_filtered.loc[df_filtered["dike_section"] == section, "init_sig_b"] = \
            df_section.loc[df_section["age_diff"] == 0., "sig_b"].mean()
        df_filtered.loc[df_filtered["dike_section"] == section, "sig_b_norm"] = \
                df_section["sig_b"] / df_section.loc[df_section["age_diff"] == 0., "sig_b"].mean()
    df_filtered["age_diff"] = df_filtered["age_diff"].replace(9999, 0)
    df_filtered["oldest_in_section"] = df_filtered["age_diff"] == 0.
    columns_keep = ["dike_section", "dike_name", "oldest_in_section", "age", "age_diff", "sig_b", "init_sig_b", "sig_b_norm"]
    df_filtered = df_filtered[columns_keep]
    return df_filtered


def find_section_groups(df: pd.DataFrame) -> List[str]:
    df = df.set_index("dike_name")
    n_nps = pd.Series(index=df.index)
    for idx, row in df.iterrows():
        n_nps.loc[idx] = len(df.columns) - row.isna().sum()
    dike_sections = list(n_nps.index[n_nps > 1])
    return dike_sections


def linear_regression(X, y):
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return beta


def run_absolute_regression(df: pd.DataFrame) -> pd.DataFrame:
    X = df["age"].values
    y = df["sig_b"].values
    beta = linear_regression(X, y)
    df["sig_b_absolute_regression"] = beta[0] + X * beta[1]
    df["sig_b_absolute_regression_norm"] = df["sig_b_absolute_regression"] / df["init_sig_b"]
    return df


def run_relative_regression(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    X = df["age_diff"].values
    y = df["sig_b_norm"].values
    beta = linear_regression(X, y)
    df["sig_b_relative_regression_norm"] = beta[0] + X * beta[1]
    age_diff_grid = np.linspace(0, 1_000, 10_000)
    sig_b_relative = beta[0] + age_diff_grid * beta[1]
    if beta[1] < 0:
        age_50_perc = age_diff_grid[np.argmin(np.abs(sig_b_relative-0.5))]
    else:
        age_50_perc = np.nan
    return df, age_50_perc


def plot_section_regression(section_res: Dict[str, Dict[str, pd.DataFrame | float]], path: Path) -> None:

    path = path / "plots"
    path.mkdir(parents=True, exist_ok=True)

    df = pd.concat([val["df"] for val in section_res.values()])
    ages_50_perc = np.array([val["age_50_perc"] for val in section_res.values()])

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="age", y="sig_b", hue="dike_section", ax=ax)
    sns.lineplot(data=df, x="age", y="sig_b_absolute_regression", hue="dike_section", ax=ax)
    ax.set_xlabel("Age [yr]", fontsize=14)
    ax.set_ylabel("Flectural strength [kPa]", fontsize=14)
    ax.legend().set_visible(False)
    ax.grid()
    plt.close()
    fig.savefig(path/"absolute_regression.png")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="age_diff", y="sig_b_norm", hue="dike_section", ax=ax)
    sns.lineplot(data=df, x="age_diff", y="sig_b_relative_regression_norm", hue="dike_section", ax=ax)
    ax.set_xlabel("Age [yr]", fontsize=14)
    ax.set_ylabel("Flectural strength [-]", fontsize=14)
    ax.legend().set_visible(False)
    ax.grid()
    plt.close()
    fig.savefig(path/"relative_regression.png")

    fig = plt.figure()
    plt.hist(ages_50_perc, color="b", alpha=0.4, ec='black', histtype='bar')
    plt.xlabel("Age of 50% strength [yr]", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.close()
    fig.savefig(path/"age_50_perc_histogram.png")


def main() -> None:

    script_path = Path(__file__).parent
    data_path = script_path.parent / "data"
    result_path = script_path.parent / f"results/regression_analysis"
    result_path.mkdir(parents=True, exist_ok=True)

    df_base = pd.read_csv(data_path/"db_querry.csv")

    dike_connections = pd.read_excel(data_path/"dike_connections.xlsx")
    dike_sections = find_section_groups(dike_connections)

    df = filter_data(df_base, dike_sections)

    dfs = []
    section_res = {}
    for section in pd.unique(df["dike_section"]):
        df_section = df.loc[df["dike_section"] == section]
        df_section = run_absolute_regression(df_section)
        df_section, age_50_perc = run_relative_regression(df_section)
        dfs.append(df_section)
        section_res[section] = {
            "df": df_section,
            "age_50_perc": age_50_perc
        }

    df = pd.concat(dfs, axis=0)
    df.to_csv(result_path/"regression_data.csv", index=False)

    plot_section_regression(section_res, result_path)

    section_res = {
        key: {
            "df": val["df"].to_json(orient='records') ,
            "age_50_perc": None if np.isnan(val["age_50_perc"]) else val["age_50_perc"]
        } for (key, val) in section_res.items()
    }

    with open(result_path/"section_regression_results.json", "w") as f:
        json.dump(section_res, f, indent=4)


if __name__ == "__main__":

    parser = ArgumentParser()
    args = parser.parse_args()

    main()

