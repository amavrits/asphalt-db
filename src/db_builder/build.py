# Tidy and DRY implementation of database build functions

import psycopg
import pandas as pd
import json
from peewee import PostgresqlDatabase
from src.db_builder.models import *


def create_db(db_config):
    db_name = db_config["database"]
    admin_conn = psycopg.connect(
        dbname="postgres",
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"],
        autocommit=True
    )
    with admin_conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cur.fetchone()

        if exists:
            print(f"Dropping existing database '{db_name}'...")
            cur.execute("""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid();
            """, (db_name,))
            cur.execute(f"DROP DATABASE {db_name}")
            print(f"✅ Database '{db_name}' dropped.")

        cur.execute(f"CREATE DATABASE {db_name}")
        print(f"✅ Database '{db_name}' created.")

    admin_conn.close()


def create_tables(db, drop_tables=False):
    tables = [
        Dike, Project, ProjectDike, Borehole, Sample, Test, GeneralData,
        StrSampleRaw, StrSampleProcessed, StrSummary,
        FtgSampleRaw, FtgSampleProcessed, FtgSummary, StiffnessSampleRaw
    ]
    if drop_tables:
        db.drop_tables(tables, safe=True)
    db.create_tables(tables, safe=True)


def resolve_borehole(borehole_name, project_name, master_table):
    dike_name = master_table.loc[
        (master_table["project"] == project_name) &
        (master_table["borehole"] == borehole_name), "dike"
    ].item()

    project = Project.get(Project.project_name == project_name)
    dike = Dike.get(Dike.dike_name == dike_name)
    project_dike = ProjectDike.get(project=project, dike=dike)
    borehole = Borehole.get(borehole_name=borehole_name, project_dike=project_dike)
    return project, dike, project_dike, borehole


def resolve_sample(sample_name, borehole_name, project_name, master_table):
    dike_name = master_table.loc[
        (master_table["project"] == project_name) &
        (master_table["borehole"] == borehole_name), "dike"
    ].item()

    project = Project.get(Project.project_name == project_name)
    dike = Dike.get(Dike.dike_name == dike_name)
    project_dike = ProjectDike.get(project=project, dike=dike)
    borehole = Borehole.get(borehole_name=borehole_name, project_dike=project_dike)
    sample = Sample.get(sample_name=sample_name, borehole=borehole)
    return project, dike, project_dike, borehole, sample


def add_project(project_name, project_data):
    Project.get_or_create(**project_data, project_name=project_name)


def add_dike(dike_name, dike_data):
    Dike.get_or_create(**dike_data, dike_name=dike_name)


def add_projectdike(dike_name, project_name):
    dike = Dike.get(Dike.dike_name == dike_name)
    project = Project.get(Project.project_name == project_name)
    ProjectDike.get_or_create(dike=dike, project=project)


def add_borehole(borehole_name, project_name, master_table, borehole_data):
    dike_name = master_table.loc[
        (master_table["project"] == project_name) &
        (master_table["borehole"] == borehole_name), "dike"
    ].item()

    project = Project.get(Project.project_name == project_name)
    dike = Dike.get(Dike.dike_name == dike_name)
    project_dike = ProjectDike.get(project=project, dike=dike)
    Borehole.create(**borehole_data, project_dike=project_dike)


def add_sample(sample_name, borehole_name, project_name, master_table, sample_data):
    *_, borehole = resolve_borehole(borehole_name, project_name, master_table)
    Sample.create(**sample_data, sample_name=sample_name, borehole=borehole)


def add_sample_general_data(sample_name, borehole_name, project_name, master_table, general_data):
    *_, sample = resolve_sample(sample_name, borehole_name, project_name, master_table)
    e = float(general_data.loc[
        (general_data["project"] == project_name) &
        (general_data["borehole"] == borehole_name) &
        (general_data["sample"] == sample_name), "e"
    ].item())
    GeneralData.create(sample=sample, e=e)


def add_sample_test(test_name, sample_name, borehole_name, project_name, master_table, borehole_folder):
    has_str = any((borehole_folder / "strength").iterdir())
    has_ftg = any((borehole_folder / "fatigue").iterdir())
    has_stf = any((borehole_folder / "stiffness").iterdir())
    *_, sample = resolve_sample(sample_name, borehole_name, project_name, master_table)
    Test.create(sample=sample, test_name=test_name, strength=has_str, fatigue=has_ftg, stiffness=has_stf)


def add_test_data(model_cls, test_name, sample_name, borehole_name, project_name, master_table, test_data: list[dict]):
    *_, sample = resolve_sample(sample_name, borehole_name, project_name, master_table)
    test = Test.get(test_name=test_name, sample=sample)
    for row in test_data:
        model_cls.create(test=test, sample_name=sample_name, **row)


def add_processed_data(model_cls, raw_cls, test_name, sample_name, borehole_name, project_name, master_table, test_data):
    *_, sample = resolve_sample(sample_name, borehole_name, project_name, master_table)
    test = Test.get(test_name=test_name, sample=sample)
    sample_raw = raw_cls.get(test=test, sample_name=sample_name)
    for row in test_data:
        model_cls.create(sample_raw=sample_raw, sample_name=sample_name, **row)


def add_summarized_data(model_cls, test_name, sample_name, borehole_name, project_name, master_table, test_data):
    *_, sample = resolve_sample(sample_name, borehole_name, project_name, master_table)
    test = Test.get(test_name=test_name, sample=sample)

    for row in test_data:  # In principle only one row
        model_cls.create(test=test, sample_name=sample_name, **row)


def iter_dikes(project_name, master_table, dike_table):
    project_dikes = master_table.loc[master_table["project"] == project_name, "dike"].unique()
    for dike_name in project_dikes:
        dike_data = dike_table.loc[dike_name, :].to_dict()
        add_dike(dike_name, dike_data)
        add_projectdike(dike_name, project_name)


def add_samples(borehole_name, project_name, master_table, test_folder, data_type="raw"):
    df = pd.read_csv(test_folder / f"{data_type}_data.csv", index_col="sample_name")
    # with open(test_folder / "test_data.json", "r") as f:
    #     test_data = json.load(f)
    for sample_name in df.index.unique():
        print(borehole_name, sample_name)
        if len(df) == 1:
            data = [df.loc[sample_name].to_dict()]
        else:
            data = df.loc[sample_name].to_dict("records")
        test_name = f"T_{sample_name}"

        if test_folder.stem == "strength":
            if data_type == "raw":
                print("Adding raw strength data strength")
                add_test_data(StrSampleRaw, test_name, sample_name, borehole_name, project_name, master_table, data)
            elif data_type == "processed":
                print("Adding processed strength data strength")
                add_processed_data(StrSampleProcessed, StrSampleRaw, test_name, sample_name, borehole_name, project_name, master_table, data)
            elif data_type == "summarized":
                print("Adding strength summary data")
                add_summarized_data(StrSummary, test_name, sample_name, borehole_name, project_name, master_table, data)
        elif test_folder.stem == "fatigue":
            if data_type == "raw":
                print("Adding raw fatigue data")
                add_test_data(FtgSampleRaw, test_name, sample_name, borehole_name, project_name, master_table, data)
            elif data_type == "processed":
                print("Adding processed fatigue data")
                add_processed_data(FtgSampleProcessed, FtgSampleRaw, test_name, sample_name, borehole_name, project_name, master_table, data)
            elif data_type == "summarized":
                print("Adding fatigue summary data")
                add_summarized_data(FtgSummary, test_name, sample_name, borehole_name, project_name, master_table, data)
        elif test_folder.stem == "stiffness":
            if data_type == "raw":
                print("Adding raw stiffness data")
                add_test_data(StiffnessSampleRaw, test_name, sample_name, borehole_name, project_name, master_table, data)
        else:
            raise ValueError(f"Unknown test type {test_folder.stem}")


if __name__ == "__main__":

    pass

