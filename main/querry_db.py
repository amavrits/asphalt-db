from src.db_builder.models import (
    Dijk, Project, ProjectDijk, Borehole, Sample, Test,
    GeneralData, BezwijksterkteRuwe, VermoeiingRuwe, StijfheidRuwe, BezwijksterkteProcessed, VermoeiingProcessed
)
from src.config import DB_CONFIG
from peewee import PostgresqlDatabase


if __name__ == "__main__":

    db = PostgresqlDatabase(**DB_CONFIG)
    db.connect()

    models = [
        Dijk, Project, ProjectDijk, Borehole, Sample, Test,
        GeneralData, BezwijksterkteRuwe, VermoeiingRuwe, StijfheidRuwe,
        BezwijksterkteProcessed, VermoeiingProcessed
    ]

    for model in models:
        count = model.select().count()
        print(f"{model.__name__}: {count} rows")

    db.close()
