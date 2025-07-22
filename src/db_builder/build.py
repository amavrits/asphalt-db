import psycopg
from peewee import PostgresqlDatabase, Model, CharField, ForeignKeyField
from src.db_builder.models import *


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


def iter_dikes(project_name, master_table, dike_table):

    project_master_table = master_table.loc[master_table["project"] == project_name]

    dike_names = project_master_table["dike"].tolist()

    for dike_name in dike_names:
        dike_data = dike_table.loc[dike_name, :]
        add_dike(dike_name, dike_data)

        add_projectdike(dike_name, project_name)




if __name__ == "__main__":

    pass

