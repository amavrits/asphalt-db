import pandas as pd
from pathlib import Path
import json
from peewee import SqliteDatabase
from src.db_builder import models
from src.db_builder.build import create_db_sqlite, create_tables, add_project, iter_dikes, add_borehole, add_sample, \
    add_sample_general_data, add_sample_test, add_samples
from src.db_builder.utils import parse_base_data

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent
    data_path = SCRIPT_DIR.parent.joinpath("data", "DATA_all")
    # r'c:\Users\hauth\OneDrive - Stichting Deltares\projects\Asphalte Regression\DB\data_all_Infram')  # make the path a env variable

    dike_table, project_table, master_table, general_data = parse_base_data(data_path)

    DB_CONFIG = {
        "engine": "sqlite",
        "path": str(Path(r'c:\Users\hauth\OneDrive - Stichting Deltares\projects\Asphalte Regression\DB').joinpath("database_all_v5 - Copy.sqlite3"))
    }

    db = SqliteDatabase(DB_CONFIG["path"])
    create_db_sqlite(DB_CONFIG)

    # Bind models to this database
    db.bind([
        models.Project,
        models.Borehole,
        models.Sample,
        models.Test,
        # ... include all model classes
    ])

    db.connect()
    create_tables(db)

    for project_folder in data_path.iterdir():
        if project_folder.is_file():
            continue

        print(f"======= PROJECT {project_folder.stem}==========")
        project_name = project_folder.stem
        project_data = project_table.loc[project_name, :]
        add_project(project_name, project_data)
        iter_dikes(project_name, master_table, dike_table)

        for borehole_folder in project_folder.iterdir():
            if borehole_folder.is_file():
                continue

            borehole_name = borehole_folder.stem
            with open(borehole_folder / "borehole_data.json", "r") as f:
                borehole_data = json.load(f)
            add_borehole(borehole_name, project_name, master_table, borehole_data)

            with open(borehole_folder / "sample_data.json", "r") as f:
                sample_data = json.load(f)

            for (sample_name, data) in sample_data.items():
                add_sample(sample_name, borehole_name, project_name, master_table, data)
                add_sample_general_data(sample_name, borehole_name, project_name, master_table, general_data)
                test_name = f"T_{sample_name}"
                add_sample_test(test_name, sample_name, borehole_name, project_name, master_table, borehole_folder)

            test_folder_list = [file for file in borehole_folder.iterdir() if file.suffix != ".json"]
            for test_folder in test_folder_list:
                for data_type in ["raw", "processed", "summarized"]:
                    if test_folder.stem == "stiffness" and data_type == "processed":
                        continue
                    add_samples(borehole_name, project_name, master_table, test_folder, data_type)

    print("Tables in DB:", db.get_tables())
    db.close()
