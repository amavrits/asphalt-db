import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import shutil


if __name__ == "__main__":

    n = 100
    n_dikes = 3
    n_projects = 1
    n_bhs = 8
    n_samples = 10

    #TODO: change the folder structure so that we remove the folder strength/fatigue/stiffness and have files raw_strength.csv, processed_strength.csv, etc.

    SCRIPT_DIR = Path(__file__).parent
    base_folder = SCRIPT_DIR.parent / "data/dummy2"
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
            master_df.loc[len(master_df)] = [f"P_{project}", f"BH_{bh}", f"D_{dike}"]

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

                sample_data = {}
                for sample in range(1, n_samples + 1):

                    general_data.append([f"P_{project}", f"BH_{bh}", f"S_{sample}", 0])

                    sample_data[f"S_{sample}"] = {
                        "depth": 0,
                        "notes": ["DDDDDD"],
                    }

                with open(borehole_path/"sample_data.json", "w") as f:
                    json.dump(sample_data, f, indent=4)

                for (test_name, test_columns) in test.items():

                    test_path = borehole_path / f"{test_name}"
                    test_path.mkdir(exist_ok=True, parents=True)

                    test_data = {
                        "str_appratus": "A",
                        "ftg_appratus": "B",
                        "stiff_appratus": "C",
                        "notes": ["DDDDDD"],
                    }

                    with open(test_path/"test_data.json", "w") as f:
                        json.dump(test_data, f, indent=4)

                    test_data = {"sample_name": [f"S_{i}" for i in range(1, n_samples+1)]}
                    for col in test_columns:
                        test_data.update({col: np.zeros(n_samples)})
                    test_data.update({"notes": "EEEEEEE"})
                    df = pd.DataFrame(test_data)
                    df.to_csv(test_path/f"{data_type}_data.csv", index=False)

    general_data = pd.DataFrame(general_data, columns=["project", "borehole", "sample", "e"])
    general_data = general_data.drop_duplicates(subset=["project", "borehole", "sample"])
    general_data.to_csv(base_folder.joinpath("general_data.csv"), index=False)



    master_df.to_csv(base_folder.joinpath("master_table.csv"), index=False)

    dike_data = {
        "dike_name": [f"D_{i}" for i in range(1, n_dikes+1)],
        "waterboard": ["HHNK"] * n_dikes,
        "notes": ["AAAA"] * n_dikes,
    }
    df_dikes = pd.DataFrame(data=dike_data)
    df_dikes.to_csv(base_folder.joinpath("dike_table.csv"), index=False)

    project_data = {
        "project_name": [f"P_{i}" for i in range(1, n_projects+1)],
        "project_code": ["BBBBB"] * n_projects,
        "date": [str(datetime.utcnow())] * n_projects,
        "notes": ["AAAA"] * n_projects
    }
    df_projects = pd.DataFrame(data=project_data)
    # df_projects.to_csv(SCRIPT_DIR.parent / f"data/dummy/project_table.csv", index=False)
    df_projects.to_csv(base_folder.joinpath("project_table.csv"), index=False)

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

