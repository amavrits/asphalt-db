import pandas as pd
import numpy as np
from pathlib import Path
import json


if __name__ == "__main__":

    n = 100
    n_dikes = 3
    n_projects = 4
    n_bhs = 5
    n_samples = 10

    SCRIPT_DIR = Path(__file__).parent

    general_data = []
    for data_type in ["raw", "processed", "summarized"]:

        with open(SCRIPT_DIR.parent/f"data/{data_type}_columns.json", "r") as f:
            test = json.load(f)

        for (test_name, test_columns) in test.items():

            for dike in range(1, n_dikes+1):

                for project in range(1, n_projects+1):

                    for bh in range(1, n_bhs+1):

                        data_path = SCRIPT_DIR.parent / f"data/dummy/{data_type}_data/{test_name}/{dike}/{project}/{bh}"
                        data_path.mkdir(exist_ok=True, parents=True)

                        for sample in range(1, n_samples+1):
                            test_data = np.random.normal(size=(n, len(test_columns)))
                            df = pd.DataFrame(test_data, columns=test_columns)
                            df.to_csv(data_path/f"{sample}.csv", index=False)

                            general_data.append([dike, project, bh, sample, np.random.uniform(0, 1)])

    general_data = pd.DataFrame(general_data, columns=["dike", "project", "borehole", "sample", "e"])
    general_data.to_csv(SCRIPT_DIR.parent / f"data/dummy/general_data.csv")
