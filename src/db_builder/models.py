from peewee import *
from playhouse.postgres_ext import PostgresqlExtDatabase
from src.config import DB_CONFIG

# Initialize DB instance
db = PostgresqlExtDatabase(**DB_CONFIG)  # TODO: Bad practice?


class BaseModel(Model):
    class Meta:
        database = db

class Dike(BaseModel):
    dike_name = CharField()
    waterboard = CharField()
    notes = TextField(null=True)

class Project(BaseModel):
    project_name = CharField()
    project_code = CharField()
    date = DateField(null=True)
    notes = TextField(null=True)

class ProjectDike(BaseModel):
    project = ForeignKeyField(Project, backref="dike_links")
    dike = ForeignKeyField(Dike, backref="project_links")

class Borehole(BaseModel):
    borehole_name = CharField()
    project_dike = ForeignKeyField(ProjectDike, backref="boreholes")
    collection_date = DateField(null=True)
    X_coord = FloatField(null=True)
    Y_coord = FloatField(null=True)
    notes = TextField(null=True)

class Sample(BaseModel):
    borehole = ForeignKeyField(Borehole, backref='samples', null=True)
    sample_name = CharField()
    depth = FloatField(null=True)
    notes = TextField(null=True)

class Test(BaseModel):
    sample = ForeignKeyField(Sample, backref='tests', null=True)
    test_name = CharField()
    notes = TextField(null=True)
    strength = BooleanField(null=False)
    fatigue = BooleanField(null=False)
    stiffness = BooleanField(null=False)

class GeneralData(BaseModel):
    sample = ForeignKeyField(Sample, backref='gen_samples', null=True)

    # Additional test fields
    e = FloatField(null=True)        # void ratio (e)

class StrSampleRaw(BaseModel):
    test = ForeignKeyField(Test, backref='str_samples', null=True)
    notes = TextField(null=True)
    sample_name = CharField()

    # STR raw test fields
    t = FloatField(null=True)        # time or thickness (depends on context)
    F = FloatField(null=True)        # force
    V_org = FloatField(null=True)    # original volume

class StrSampleProcessed(BaseModel):
    sample_raw = ForeignKeyField(StrSampleRaw, backref='processed_samples', null=True)
    notes = TextField(null=True)
    sample_name = CharField()

    # STR processed test fields
    F = FloatField(null=True)         # force
    V_cor = FloatField(null=True)     # corrected volume
    eps = FloatField(null=True)       # strain
    sig = FloatField(null=True)       # stress
    Sec = FloatField(null=True)       # secant modulus

class StrSampleSummary(BaseModel):
    sample_processed = ForeignKeyField(StrSampleProcessed, backref='summarized_samples', null=True)
    sample_name = CharField()
    str = FloatField()

class FtgSampleRaw(BaseModel):
    test = ForeignKeyField(Test, backref='ftg_samples', null=True)
    notes = TextField(null=True)
    sample_name = CharField()

    # FTG-specific test fields
    N = IntegerField(null=True)  # Number of cycles, perhaps?
    maximum_stroke = FloatField(null=True)
    minimum_stroke = FloatField(null=True)
    peak_to_peak_stroke = FloatField(null=True)
    maximum_load = FloatField(null=True)
    peak_to_peak_load = FloatField(null=True)
    in_phase_modulus = FloatField(null=True)
    out_phase_modulus = FloatField(null=True)

class FtgSampleProcessed(BaseModel):
    sample_raw = ForeignKeyField(FtgSampleRaw, backref='processed_samples', null=True)
    notes = TextField(null=True)
    sample_name = CharField()

    # FTG processed test fields
    N = IntegerField(null=True)
    eps_cycl = FloatField(null=True)   # cyclic strain
    eps_perm = FloatField(null=True)   # permanent strain
    sig_cyc = FloatField(null=True)    # cyclic stress
    sig_perm = FloatField(null=True)   # permanent stress
    E_dyn = FloatField(null=True)      # dynamic modulus
    pha = FloatField(null=True)        # phase angle

class StiffnessSampleRaw(BaseModel):
    test = ForeignKeyField(Test, backref='Edyn_samples', null=True)
    notes = TextField(null=True)
    sample_name = CharField()

    # Edyn raw test fields
    T = FloatField(null=True)         # Temperature (assumed)
    f = FloatField(null=True)         # Frequency
    eps = FloatField(null=True)       # Strain
    E_dyn = FloatField(null=True)     # Dynamic modulus
    pha = FloatField(null=True)       # Phase angle


if __name__ == "__main__":

    pass

