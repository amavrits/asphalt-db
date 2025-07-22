import psycopg
from peewee import PostgresqlDatabase, Model, CharField, ForeignKeyField
from src.db_builder.models import *
import pandas as pd
import json


def create_db(db_config):

    admin_conn = psycopg.connect(
        dbname="postgres",  # connect to default DB to create your target DB
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"],
        autocommit=True
    )

    with admin_conn.cursor() as cur:

        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_config["database"],))
        exists = cur.fetchone()

        if not exists:
            cur.execute(f"CREATE DATABASE {db_config['database']}")
            print(f"Database '{db_config['database']}' created.")
        else:
            print(f"Database '{db_config['database']}' already exists.")

    admin_conn.close()


def create_tables(db):

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


def add_project(project_name, project_data):
    Project.get_or_create(
        project_name=project_name,
        project_code=project_data["project_code"],
        date=project_data["date"],
        notes=project_data["notes"]
    )


def add_dike(dike_name, dike_data):
    Dike(
        dike_name=dike_name,
        waterboard=dike_data["waterboard"],
        notes=dike_data["notes"]
    )


def add_projectdike(dike_name, project_name):
    ProjectDike(
        dike=Dike(dike_name=dike_name),
        project=Project(project_name=project_name)
    )


def add_borehole(borehole_name, project_name, master_table, borehole_data):

    cond = (master_table["project"] == project_name) & (master_table["borehole"] == borehole_name)
    dike_name = master_table.loc[cond, "dike"].item()

    Borehole(
        borehole_name=borehole_name,
        project_dike=ProjectDike(
            dike=Dike(dike_name=dike_name),
            project=Project(project_name=project_name)
        ),
        collection_date=borehole_data["collection_date"],
        X_coord=borehole_data["X_coord"],
        Y_coord=borehole_data["Y_coord"],
        notes=borehole_data["notes"]
    )


def add_sample(sample_name, borehole_name, sample_data):
    Sample(
        borehole=Borehole(borehole_name=borehole_name),
        sample_name=sample_name,
        depth=sample_data["depth"],
        notes=sample_data["notes"]
    )


def add_sample_general_data(sample_name, borehole_name, project_name, general_data):

    cond = (general_data["project"] == project_name) & \
           (general_data["borehole"] == borehole_name) & \
           (general_data["sample"] == sample_name)
    general_data_sample = general_data.loc[cond, :]

    GeneralData(
        sample=Sample(sample_name=sample_name),
        e=general_data_sample["e"]
    )


def add_sample_test(test_name, sample_name, borehole_folder):

    has_str = not any((borehole_folder / "str").iterdir())
    has_ftg = not any((borehole_folder / "ftg").iterdir())
    has_stf = not any((borehole_folder / "stf").iterdir())

    Test(
        sample=Sample(sample_name=sample_name),
        test_name=test_name,
        strength=has_str,
        fatigue=has_ftg,
        stiffness=has_stf,
    )


def add_str_raw(test_name, test_data):
    StrSampleRaw(
        test=Test(test_name=test_name),
        notes=test_data["notes"],
        t=test_data["t"],
        F=test_data["F"],
        V_org=test_data["V_org"],
    )


def add_ftg_raw(test_name, test_data):
    FtgSampleRaw(
        test=Test(test_name=test_name),
        notes=test_data["notes"],
        N=test_data["N"],
        maximum_stroke=test_data["MaximumStroke"],
        minimum_stroke=test_data["MinimumStroke"],
        peak_to_peak_stroke=test_data["PeakToPeakStroke"],
        maximum_load=test_data["MaximumLoad"],
        peak_to_peak_load=test_data["PeakToPeakLoad"],
        in_phase_modulus=test_data["InPhaseModulus"],
        out_phase_modulus=test_data["OutPhaseModulus"],
    )


def add_edyn_raw(test_name, test_data):
    EdynSampleRaw(
        test=Test(test_name=test_name),
        notes=test_data["notes"],
        T=test_data["T"],
        f=test_data["f"],
        eps=test_data["eps"],
        E_dyn=test_data["E_dyn"],
        pha=test_data["pha"]
    )



def add_str_processed(test_name, test_data):
    StrSampleProcessed(
        sample_raw=StrSampleRaw(test_name=test_name),
        notes=test_data["notes"],
        F=test_data["F"],
        V_cor=test_data["V_cor"],
        eps=test_data["eps"],
        sig=test_data["sig"],
        Sec=test_data["Sec"],
    )


def add_ftg_processed(test_name, test_data):
    FtgSampleProcessed(
        sample_raw=FtgSampleRaw(test_name=test_name),
        notes=test_data["notes"],
        N=test_data["N"],
        eps_cycl=test_data["eps_cycl"],
        eps_perm=test_data["eps_perm"],
        sig_cyc=test_data["sig_cyc"],
        sig_perm=test_data["sig_perm"],
        E_dyn=test_data["E_dyn"],
        pha=test_data["pha"],
    )


def iter_dikes(project_name, master_table, dike_table):

    project_master_table = master_table.loc[master_table["project"] == project_name]

    dike_names = project_master_table["dike"].tolist()

    for dike_name in dike_names:
        dike_data = dike_table.loc[dike_name, :]
        add_dike(dike_name, dike_data)

        add_projectdike(dike_name, project_name)


def add_samples(test_name, test_folder, data_type="raw"):

    df = pd.read_csv(test_folder / f"{data_type}_data.csv", index_col="sample_name")

    with open(test_folder / "test_data.json", "r") as f:
        test_data = json.load(f)

    for sample_name in df.index:

        data = df.loc[sample_name]

        if test_folder.stem == "str":

            if data_type == "raw":
                add_str_raw(test_name, data)
            elif data_type == "processed":
                add_str_processed(test_name, data)
            else:
                raise ValueError(f"Unknown data type: {data_type}")

        elif test_folder.stem == "ftg":

            if data_type == "raw":
                add_ftg_raw(test_name, data)
            elif data_type == "processed":
                add_ftg_processed(test_name, data)
            else:
                raise ValueError(f"Unknown data type: {data_type}")

        elif test_folder.stem == "stf":

            if data_type == "processed":
                continue

            if data_type == "raw":
                add_edyn_raw(test_name, data)
            else:
                raise ValueError(f"Unknown data type: {data_type}")

        else:

            raise ValueError(f"Unknown test type {test_folder.stem}")


if __name__ == "__main__":

    pass

