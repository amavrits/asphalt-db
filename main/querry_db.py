from src.db_builder.models import (
    Dike, Project, ProjectDike, Borehole, Sample, Test,
    GeneralData, StrSampleRaw, FtgSampleRaw, EdynSampleRaw
)
from src.config import DB_CONFIG
from peewee import PostgresqlDatabase


if __name__ == "__main__":

    db = PostgresqlDatabase(**DB_CONFIG)
    db.connect()

    models = [
        Dike, Project, ProjectDike, Borehole, Sample, Test,
        GeneralData, StrSampleRaw, FtgSampleRaw, EdynSampleRaw
    ]

    for model in models:
        count = model.select().count()
        print(f"{model.__name__}: {count} rows")

    db.close()
