from peewee import *
from playhouse.postgres_ext import PostgresqlExtDatabase
from src.config import DB_CONFIG

# Initialize DB instance
db = PostgresqlExtDatabase(**DB_CONFIG)  # TODO: Bad practice?


class BaseModel(Model):
    class Meta:
        database = db

class Dijk(BaseModel):
    dike_name = CharField()
    waterboard = CharField()
    notes = TextField(null=True)

class Project(BaseModel):
    project_name = CharField()
    project_code = CharField()
    date = DateField(null=True)
    notes = TextField(null=True)

class ProjectDijk(BaseModel):
    project = ForeignKeyField(Project, backref="dijk_links")
    dijk = ForeignKeyField(Dijk, backref="project_links")

class Borehole(BaseModel):
    borehole_name = CharField()
    project_dijk = ForeignKeyField(ProjectDijk, backref="boreholes")
    collection_date = DateField(null=True)
    aanlegjaar = IntegerField(null=True)  # Construction year
    onderzoeksjaar = IntegerField(null=True)  # Sample year
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
    HR = FloatField(null=True)        # void
    bitumen = FloatField(null=True)  # Bitumen content
    dichtheid = FloatField(null=True)  # Density

class BezwijksterkteRuwe(BaseModel):
    test = ForeignKeyField(Test, backref='strength_samples', null=True)
    notes = TextField(null=True)
    sample_name = CharField()

    # STR raw test fields
    t = FloatField(null=True)        #
    F = FloatField(null=True)        # force
    V_org = FloatField(null=True)    # original volume

class BezwijksterkteProcessed(BaseModel):
    sample_raw = ForeignKeyField(BezwijksterkteRuwe, backref='strength_processed_samples', null=True)
    notes = TextField(null=True)
    sample_name = CharField()

    # STR processed test fields
    F = FloatField(null=True)         # force
    V_cor = FloatField(null=True)     # corrected volume
    eps = FloatField(null=True)       # strain
    sig = FloatField(null=True)       # stress
    Sec = FloatField(null=True)       # secant modulus

class BezwijksterkteSummary(BaseModel):
    test = ForeignKeyField(Test, backref='strength_samples_summary', null=True)
    sample_name = CharField()
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



class VermoeiingRuwe(BaseModel):
    test = ForeignKeyField(Test, backref='fatigue_samples', null=True)
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

class VermoeiingProcessed(BaseModel):
    sample_raw = ForeignKeyField(VermoeiingRuwe, backref='fatigue_processed_samples', null=True)
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

class VermoeiingSummary(BaseModel):
    test = ForeignKeyField(Test, backref='fatigue_samples_summary', null=True)
    sample_name = CharField()
    pha_ini = FloatField()  # Initial phase angle
    pha_50 = FloatField()  # Phase angle at 50% of cycles
    sig_cyc = FloatField()  # Cyclic stress
    sig_perm = FloatField()  # Permanent stress
    E_ini = FloatField()  # Initial dynamic modulus
    E_50 = FloatField()  # Dynamic modulus at 50% of cycles
    N_fat = IntegerField()  # Number of fatigue cycles


class StijfheidRuwe(BaseModel):
    test = ForeignKeyField(Test, backref='Edyn_samples', null=True)
    notes = TextField(null=True)
    sample_name = CharField()

    # Edyn raw test fields
    T = FloatField(null=True)         # Temperature (assumed)
    f = FloatField(null=True)         # Frequency
    eps = FloatField(null=True)       # Strain
    E_dyn = FloatField(null=True)     # Dynamic modulus
    pha = FloatField(null=True)       # Phase angle

class StijfheidSummary(BaseModel):
    test = ForeignKeyField(Test, backref='stf_samples_summary', null=True)
    notes = TextField(null=True)
    sample_name = CharField()

    temp = FloatField(null=True)  # Temperature
    E_dyn_value = FloatField(null=True)  # Dynamic modulus value

if __name__ == "__main__":

    pass

