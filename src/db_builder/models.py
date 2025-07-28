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
    thickness = FloatField(null=True)  # Thickness of the sample
    height = FloatField(null=True)
    strength = FloatField(null=True)  # find a better name for this field
    v = FloatField(null=True)  # find a better name for this field
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

class StrSummary(BaseModel):
    test = ForeignKeyField(Test, backref='str_samples_summary', null=True)
    sample_name = CharField()

    HR = FloatField()
    v = FloatField()
    Sec_10 = FloatField()
    Sec_50 = FloatField()
    Sec_100 = FloatField()
    sig_b = FloatField()
    eps_b = FloatField()
    G_c = FloatField()
    G_c_over_eps_b = FloatField()
    G_c_over_eps_b_sig_b = FloatField()
    V_Ber = FloatField()



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

class FtgSummary(BaseModel):
    test = ForeignKeyField(Test, backref='ftg_samples_summary', null=True)
    sample_name = CharField()
    pha_ini = FloatField()  # Initial phase angle
    pha_50 = FloatField()  # Phase angle at 50% of cycles
    sig_cyc = FloatField()  # Cyclic stress
    sig_perm = FloatField()  # Permanent stress
    E_ini = FloatField()  # Initial dynamic modulus
    E_50 = FloatField()  # Dynamic modulus at 50% of cycles
    N_fat = IntegerField()  # Number of fatigue cycles


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

