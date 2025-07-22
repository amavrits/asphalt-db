from src.parsing.parse import *
import pandas as pd
from pathlib import Path


if __name__ == "__main__":

    SCRIPT_DIR = Path(__file__).parent
    base_data_path = SCRIPT_DIR.parent / "data/dummy"

    general_data = pd.read_csv(base_data_path/"general_data.csv")

    dfs = []
    for test_folder in (base_data_path/"raw_data").iterdir():
        test_name = test_folder.stem

        for dike_folder in test_folder.iterdir():
            dike_name = int(dike_folder.stem)

            for project_folder in dike_folder.iterdir():
                project_name = int(project_folder.stem)

                for bh_folder in project_folder.iterdir():
                    bh_name = int(bh_folder.stem)

                    for sample in bh_folder.glob("*.csv"):

                        sample_name = int(sample.stem)
                        df = pd.read_csv(sample)
                        df[["dike_name", "project_name", "project_code", "borehole_name", "sample_name"]] = [dike_name, project_name, project_name, bh_name, sample_name]
                        dfs.append(df)


    df = pd.concat(dfs, axis=0)
    df.to_csv(base_data_path/"compiled_data.csv", index=False)
