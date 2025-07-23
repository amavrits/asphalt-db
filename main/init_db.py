import pandas as pd
from src.config import DB_CONFIG
from src.db_builder.models import *
from src.db_builder.build import *
from src.db_builder.utils import *
from pathlib import Path
import json


if __name__ == "__main__":

    SCRIPT_DIR = Path(__file__).parent
    data_path = SCRIPT_DIR.parent / "data/dummy_small"

    dike_table, project_table, master_table, general_data = parse_base_data(data_path)

    create_db(DB_CONFIG)

    db.connect()

    create_tables(db)

    for project_folder in data_path.iterdir():

        if project_folder.is_file():
            continue

        project_name = project_folder.stem

        project_data = project_table.loc[project_name, :]
        add_project(project_name, project_data)

        iter_dikes(project_name, master_table, dike_table)

        for borehole_folder in project_folder.iterdir():

            if borehole_folder.is_file():
                continue

            borehole_name = borehole_folder.stem

            with open(borehole_folder/"borehole_data.json", "r") as f:
                borehole_data = json.load(f)

            add_borehole(borehole_name, project_name, master_table, borehole_data)

            with open(borehole_folder / "sample_data.json", "r") as f:
                sample_data = json.load(f)

            for (sample_name, data) in sample_data.items():

                add_sample(sample_name, borehole_name, project_name, master_table, data)

                add_sample_general_data(sample_name, borehole_name, project_name, master_table, general_data)

                test_name = f"T_{sample_name}"
                add_sample_test(test_name, sample_name, borehole_name, project_name, master_table, borehole_folder)

            test_folder_list = [file for file in borehole_folder.iterdir() if file.name.split(".")[-1] != "json"]

            for test_folder in test_folder_list:

                for data_type in ["raw", "processed"]:
                    add_samples(borehole_name, project_name, master_table, test_folder, data_type)

    db.close()

