import pandas as pd
from pathlib import Path

def parse_base_data(data_path):

    if not isinstance(data_path, Path):
        data_path = Path(data_path)

    dike_table = pd.read_csv(data_path/"dike_table.csv", index_col="dike_name")
    project_table = pd.read_csv(data_path/"project_table.csv", index_col="project_name")
    master_table = pd.read_csv(data_path/"master_table.csv")
    general_data = pd.read_csv(data_path/"general_data.csv")

    return dike_table, project_table, master_table, general_data


if __name__ == "__main__":

    pass

