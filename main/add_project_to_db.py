import json
from pathlib import Path

from peewee import SqliteDatabase

from src.db_builder import models
from src.db_builder.build import add_project, iter_dikes, add_borehole, add_sample, add_sample_general_data, \
    add_sample_test, add_samples
from src.db_builder.utils import parse_base_data

SCRIPT_DIR = Path(__file__).parent
project_folder = SCRIPT_DIR.parent.joinpath("data", "DATA_1400863")
project_name = "P_1400863"


dike_table, project_table, master_table, general_data = parse_base_data(project_folder)

######################## ATTENTION !! MAKE SUR YOU FIRST MAKE A BACKUP OF YOUR DATABASE ########################
######################## ATTENTION !! MAKE SUR YOU FIRST MAKE A BACKUP OF YOUR DATABASE ########################
######################## ATTENTION !! MAKE SUR YOU FIRST MAKE A BACKUP OF YOUR DATABASE ########################
######################## ATTENTION !! MAKE SUR YOU FIRST MAKE A BACKUP OF YOUR DATABASE ########################
######################## ATTENTION !! MAKE SUR YOU FIRST MAKE A BACKUP OF YOUR DATABASE ########################
######################## ATTENTION !! MAKE SUR YOU FIRST MAKE A BACKUP OF YOUR DATABASE ########################


DB_PATH = Path(
        r"c:\Users\hauth\OneDrive - Stichting Deltares\projects\Asphalte Regression\DB\database_all_v3.sqlite3"
    )

# === CONNECT TO EXISTING DATABASE ===
db = SqliteDatabase(str(DB_PATH))

# Bind models to this database
db.bind([
    models.Project,
    models.Borehole,
    models.Sample,
    models.Test,
    # add any other model classes here if necessary
])

db.connect(reuse_if_open=True)

# Create tables only if they don’t exist
db.create_tables([
    models.Project,
    models.Borehole,
    models.Sample,
    models.Test,
], safe=True)

if not project_folder.exists():
    raise FileNotFoundError(f"Project folder {project_folder} not found")

# Add just this project
project_data = project_table.loc[project_name, :]
add_project(project_name, project_data)
iter_dikes(project_name, master_table, dike_table)

project_folder = project_folder.joinpath(project_name)
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



# === WRAP UP ===
print("\n✅ Project added successfully!")
print("📋 Tables in DB:", db.get_tables())

db.close()
print("🔒 Database connection closed.")