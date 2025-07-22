import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import shutil


if __name__ == "__main__":

    n = 100
    n_dikes = 3
    n_projects = 4
    n_bhs = 5
    n_samples = 10

    SCRIPT_DIR = Path(__file__).parent
    base_folder = SCRIPT_DIR.parent / "data/dummy"
    if base_folder.is_dir():
        shutil.rmtree(base_folder)
    base_folder.mkdir(exist_ok=True, parents=True)

    general_data = []

    master_df = pd.DataFrame([], columns=["project", "borehole", "dike"])
    for i, project in enumerate(range(1, n_projects+1)):

        for j, bh in enumerate(range(1, n_bhs+1)):

            borehole_path = base_folder / f"P_{project}/BH_{bh}"
            borehole_path.mkdir(exist_ok=True, parents=True)

            np.random.seed(i * 1_000 + j)
            dike = np.random.randint(1, n_dikes+1)
            master_df.loc[len(master_df)] = [project, bh, dike]

            borehole_data = {
                "borehole_name": f"BH_{bh}",
                "collection_date": str(datetime.utcnow()),
                "notes": ["AAAA"],
                "X_coord": 0,
                "Y_coord": 0,
            }
            with open(borehole_path/"borehole_data.json", "w") as f:
                json.dump(borehole_data, f, indent=4)

            for data_type in ["raw", "processed", "summarized"]:

                with open(SCRIPT_DIR.parent / f"data/{data_type}_columns.json", "r") as f:
                    test = json.load(f)

                for sample in range(1, n_samples + 1):

                    sample_path = borehole_path / f"S_{sample}"
                    sample_path.mkdir(exist_ok=True, parents=True)

                    sample_data = {
                        "sample_name": f"S_{sample}",
                        "depth": 0,
                        "notes": ["DDDDDD"],
                    }

                    with open(sample_path/"sample_data.json", "w") as f:
                        json.dump(sample_data, f, indent=4)

                    test_data = {
                        "str_appratus": "A",
                        "ftg_appratus": "B",
                        "stiff_appratus": "C",
                        "notes": ["DDDDDD"],
                    }

                    with open(sample_path/"test_data.json", "w") as f:
                        json.dump(test_data, f, indent=4)

                    for (test_name, test_columns) in test.items():

                        test_data = np.zeros((n, len(test_columns)))
                        df = pd.DataFrame(test_data, columns=test_columns)
                        df.to_csv(sample_path/f"{test_name}.csv", index=False)

                    general_data.append([project, bh, sample, 0])

    general_data = pd.DataFrame(general_data, columns=["project", "borehole", "sample", "e"])
    general_data.to_csv(SCRIPT_DIR.parent / f"data/dummy/general_data.csv")

    master_df.to_csv(SCRIPT_DIR.parent / f"data/dummy/master_table.csv")

    dike_data = {
        "waterboard": ["HHNK"] * n_dikes,
        "notes": ["AAAA"] * n_dikes,
    }
    df_dikes = pd.DataFrame(data=dike_data)
    df_dikes.to_csv(SCRIPT_DIR.parent / f"data/dummy/dike_data.csv")

    dike_data = {
        "dike_name": [f"D_{i}" for i in range(1, n_dikes+1)],
        "waterboard": ["HHNK"] * n_dikes,
        "notes": ["AAAA"] * n_dikes,
    }
    df_dikes = pd.DataFrame(data=dike_data)
    df_dikes.to_csv(SCRIPT_DIR.parent / f"data/dummy/dike_data.csv")

    project_data = {
        "project_name": [f"P_{i}" for i in range(1, n_projects+1)],
        "project_code": ["BBBBB"] * n_projects,
        "date": [str(datetime.utcnow())] * n_projects,
        "notes": ["AAAA"] * n_projects
    }
    df_projects = pd.DataFrame(data=project_data)
    df_projects.to_csv(SCRIPT_DIR.parent / f"data/dummy/project_data.csv")

    # n_total_boreholes = len(master_df)
    # borehole_data = {
    #     "borehole_name": [f"BH_{i}" for i in master_df["borehole"]],
    #     "collection_date": [datetime.utcnow()] * n_total_boreholes,
    #     "notes": ["AAAA"] * n_total_boreholes,
    #     "X_coord": np.zeros(n_total_boreholes),
    #     "Y_coord": np.zeros(n_total_boreholes),
    # }
    # borehole_projects = pd.DataFrame(data=borehole_data)
    # borehole_projects.to_csv(SCRIPT_DIR.parent / f"data/dummy/borehole_data.csv")

