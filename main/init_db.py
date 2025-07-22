import os
import pandas as pd
from src.config import DB_CONFIG
from src.db_builder.models import *
from src.db_builder.build import *
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime


if __name__ == "__main__":

    # -----------------------
    # Load environment variables
    # -----------------------

    SCRIPT_DIR = Path(__file__).parent
    data_path = SCRIPT_DIR.parent / "data/dummy/compiled_data.csv"
    dotenv_path = SCRIPT_DIR.parent / ".env"
    load_dotenv(dotenv_path)

    df = pd.read_csv(data_path)
    df[["waterboard", "notes", "X_coord", "Y_coord", "depth", "e"]] = 0
    df[["date", "collection_date"]] = datetime.utcnow()





    # -----------------------
    # Step 1: Create the database if it doesn't exist
    # -----------------------

    create_db(DB_CONFIG)

    # -----------------------
    # Step 2: Create table and populate data
    # -----------------------

    db.connect()

    db.drop_tables([
        Dike,
        Project,
        ProjectDike,
        Borehole,
        Sample,
        Test,
        GeneralData,
        StrSampleRaw,
        FtgSampleRaw,
        EdynSampleRaw
    ],
        safe=True)

    db.create_tables([
        Dike,
        Project,
        ProjectDike,
        Borehole,
        Sample,
        Test,
        GeneralData,
        StrSampleRaw,
        FtgSampleRaw,
        EdynSampleRaw
    ],
        safe=True)



    dike_df = df.drop_duplicates(subset=["dike_name"])
    dikes = list(dike_df[["dike_name", "waterboard", "notes"]].T.to_dict().values())
    for dike_data in dikes:
        Dike.get_or_create(**dike_data)

    project_df = df.drop_duplicates(subset=["project_name"])
    projects = list(project_df[["project_name", "project_code", "date", "notes"]].T.to_dict().values())
    for project_data in projects:
        Project.get_or_create(**project_data)

    dikeproject_df = df.drop_duplicates(subset=["dike_name", "project_name"])
    for i, row in dikeproject_df.iterrows():
        dike_name, project_name = row["dike_name"], row["project_name"]
        dike, _ = Dike.get_or_create(dike_name=dike_name)
        project, _ = Project.get_or_create(project_name=project_name)
        ProjectDike.get_or_create(dike=dike, project=project)

    
    borehole_df = df.drop_duplicates(subset=["borehole_name", "dike_name", "project_name"])
    for i, row in borehole_df.iterrows():
        dike_name, project_name = row["dike_name"], row["project_name"]
        dike, _ = Dike.get_or_create(dike_name=dike_name)
        project, _ = Project.get_or_create(project_name=project_name)
        dp, _ = ProjectDike.get_or_create(dike=dike, project=project)
        Borehole.get_or_create(
            borehole_name=row["borehole_name"],
            project_dike=dp,
            collection_date=row["collection_date"],
            X_coord=row["X_coord"],
            Y_coord=row["Y_coord"],
            notes=row["notes"],
        )


    sample_df = df.drop_duplicates(subset=["sample_name", "borehole_name", "dike_name", "project_name"])
    for i, row in sample_df.iterrows():
        dike_name, project_name = row["dike_name"], row["project_name"]
        dike, _ = Dike.get_or_create(dike_name=dike_name)
        project, _ = Project.get_or_create(project_name=project_name)
        dp, _ = ProjectDike.get_or_create(dike=dike, project=project)
        bh = Borehole.get_or_create(project_dike=dp)
        Sample.get_or_create(
            borehole=bh,
            sample_name=row["sample_name"],
            depth=row["collection_date"],
            notes=row["notes"],
        )


    db.close()

