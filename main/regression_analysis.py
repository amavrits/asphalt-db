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
    df_filtered["sig_b_norm"] = 0.
    for section in pd.unique(df_filtered["dike_section"]):
        df_section = df_filtered.loc[df_filtered["dike_section"] == section]
        ages = df_section["age"].values
        unique_ages = np.unique(df_section["age"])
        age_diffs = ages[:, np.newaxis] - unique_ages
        age_diffs[age_diffs<=0] = 9999
        df_filtered.loc[df_filtered["dike_section"] == section, "age_diff"] = age_diffs.min(axis=-1)
        df_filtered.loc[df_filtered["dike_section"] == section, "sig_b_norm"] = (
                df_section["sig_b"] / df_section.loc[df_section["age_diff"] == 0., "sig_b"].mean())
    df_filtered["age_diff"] = df_filtered["age_diff"].replace(9999, 0)
    df_filtered["oldest_in_section"] = df_filtered["age_diff"] == 0.
    columns_keep = ["dike_section", "dike_name", "oldest_in_section", "age", "age_diff", "sig_b", "sig_b_norm"]
    df_filtered = df_filtered[columns_keep]
    return df_filtered


def find_section_groups(df: pd.DataFrame) -> List[str]:
    df = df.set_index("dike_name")
    n_nps = pd.Series(index=df.index)
    for idx, row in df.iterrows():
        n_nps.loc[idx] = len(df.columns) - row.isna().sum()
    dike_sections = list(n_nps.index[n_nps > 1])
    return dike_sections


def main() -> None:

    script_path = Path(__file__).parent
    data_path = script_path.parent / "data"
    result_path = script_path.parent / f"results/regression_analysis"
    result_path.mkdir(parents=True, exist_ok=True)

    df_base = pd.read_csv(data_path/"db_querry.csv")

    dike_connections = pd.read_excel(data_path/"dike_connections.xlsx")
    dike_sections = find_section_groups(dike_connections)

    df = filter_data(df_base, dike_sections)

    # with open(result_path/"bootstrap_results.json", "w") as f:
    #     json.dump(age_bootstrap_res, f, indent=4)

    pass


if __name__ == "__main__":

    parser = ArgumentParser()
    args = parser.parse_args()

    main()

