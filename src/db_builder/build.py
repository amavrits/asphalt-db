from peewee import *
from playhouse.postgres_ext import PostgresqlExtDatabase
from config import DB_CONFIG

# Initialize DB instance
db = PostgresqlExtDatabase(**DB_CONFIG)

class BaseModel(Model):
    class Meta:
        database = db

class Sample(BaseModel):
    borehole = ForeignKeyField(Borehole, backref='samples', null=True)
    sample_name = CharField()
    collection_date = DateField(null=True)
    depth = FloatField(null=True)
    notes = TextField(null=True)

class Borehole(BaseModel):
    borehole_name = CharField()
    project_dike = ForeignKeyField(ProjectDike, backref="boreholes")
    collection_date = DateField(null=True)
    location = CharField(null=True)
    notes = TextField(null=True)

class Project(BaseModel):
    project_name = CharField()
    project_id = CharField()
    date = DateField(null=True)
    notes = TextField(null=True)

class Dike(BaseModel):
    dike_name = CharField()
    waterboard = CharField()
    notes = TextField(null=True)

class ProjectDike(BaseModel):
    project = ForeignKeyField(Project, backref="dike_links")
    dike = ForeignKeyField(Dike, backref="project_links")


if __name__ == "__main__":

    pass

