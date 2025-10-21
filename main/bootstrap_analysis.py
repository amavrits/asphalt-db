import pandas as pd
import numpy as np
import random
from pathlib import Path
import json
from numpy.typing import NDArray
from typing import List, Dict, Tuple


def select_dike(dike_id: str = "emmapolder") -> str:
    available_dikes = pd.unique(df_multiple_projects["dike_name"]).tolist()
    dike_name = [d for d in available_dikes if dike_id in d]
    if len(dike_name) > 1:
        raise ValueError(f"Invalid dike ID, '{dike_id}' is present in multiple dike names {available_dikes}.")
    else:
        dike_name = dike_name[0]
    return dike_name


def main(dike_id: str = "emmapolder") -> None:

    script_path = Path(__file__).parent
    data_path = script_path.parent /"data/db_querry.csv"

    df = pd.read_csv(data_path)

    df_multiple_projects = df.groupby("dike_name").filter(lambda x: x["project_name"].nunique() >= 2)
    df_multiple_projects = df_multiple_projects.sort_values(by=["dike_name", "project_name"])



    pass


if __name__ == "__main__":

    main()

