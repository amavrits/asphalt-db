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
    borehole = ForeignKeyField(Sample, backref='tests', null=True)
    notes = TextField(null=True)
    strength = BooleanField(null=False)
    fatigue = BooleanField(null=False)
    stiffness = BooleanField(null=False)

class GeneralData(BaseModel):
    sample = ForeignKeyField(Sample, backref='gen_samples', null=True)
    notes = TextField(null=True)

    # Additional test fields
    e = FloatField(null=True)        # void ratio (e)

class StrSampleRaw(BaseModel):
    borehole = ForeignKeyField(Test, backref='str_samples', null=True)
    notes = TextField(null=True)

    # STR raw test fields
    t = FloatField(null=True)        # time or thickness (depends on context)
    F = FloatField(null=True)        # force
    V_org = FloatField(null=True)    # original volume

class StrSampleProcessed(BaseModel):
    sample_raw = ForeignKeyField(Test, backref='processed_samples', null=True)
    notes = TextField(null=True)

    # STR processed test fields
    F = FloatField(null=True)         # force
    V_cor = FloatField(null=True)     # corrected volume
    eps = FloatField(null=True)       # strain
    sig = FloatField(null=True)       # stress
    Sec = FloatField(null=True)       # secant modulus

# class StrSampleSummarized(BaseModel):
#     sample_processed = ForeignKeyField(StrSampleProcessed, backref='summarized_samples', null=True)
#     sample_name = CharField()
#     collection_date = DateField(null=True)
#     depth = FloatField(null=True)
#     notes = TextField(null=True)
#
#     # STR summarized test fields
#     HR = FloatField(null=True)                 # Hardening ratio?
#     v = FloatField(null=True)                  # Poisson's ratio?
#     sig_b = FloatField(null=True)              # Breaking stress
#     eps_b = FloatField(null=True)              # Breaking strain
#     Sec_10 = FloatField(null=True)             # Secant modulus at 10%
#     Sec_50 = FloatField(null=True)             # Secant modulus at 50%
#     Sec_100 = FloatField(null=True)            # Secant modulus at 100%
#     G_c = FloatField(null=True)                # Fracture energy?
#     G_c_over_eps_b = FloatField(null=True)     # G_c / eps_b
#     G_c_over_eps_b_sig_b = FloatField(null=True)  # G_c / (eps_b * sig_b)
#     V_Ber = FloatField(null=True)              # Some volume metric?

class FtgSampleRaw(BaseModel):
    borehole = ForeignKeyField(Test, backref='ftg_samples', null=True)
    notes = TextField(null=True)

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
    sample_raw = ForeignKeyField(Test, backref='processed_samples', null=True)
    notes = TextField(null=True)

    # FTG processed test fields
    N = IntegerField(null=True)
    eps_cycl = FloatField(null=True)   # cyclic strain
    eps_perm = FloatField(null=True)   # permanent strain
    sig_cyc = FloatField(null=True)    # cyclic stress
    sig_perm = FloatField(null=True)   # permanent stress
    E_dyn = FloatField(null=True)      # dynamic modulus
    pha = FloatField(null=True)        # phase angle

# class FtgSampleSummarized(BaseModel):
#     sample_processed = ForeignKeyField(FtgSampleProcessed, backref='summarized_samples', null=True)
#     sample_name = CharField()
#     collection_date = DateField(null=True)
#     depth = FloatField(null=True)
#     notes = TextField(null=True)
#
#     # FTG summarized test fields
#     sig_cyc = FloatField(null=True)    # cyclic stress
#     sig_perm = FloatField(null=True)   # permanent stress
#     E_ini = FloatField(null=True)      # initial modulus
#     pha_ini = FloatField(null=True)    # initial phase angle
#     E_50 = FloatField(null=True)       # modulus at 50% fatigue life
#     pha_50 = FloatField(null=True)     # phase angle at 50% fatigue life
#     N_fat = IntegerField(null=True)    # fatigue life (cycle count)

class EdynSampleRaw(BaseModel):
    borehole = ForeignKeyField(Test, backref='Edyn_samples', null=True)
    notes = TextField(null=True)

    # Edyn raw test fields
    T = FloatField(null=True)         # Temperature (assumed)
    f = FloatField(null=True)         # Frequency
    eps = FloatField(null=True)       # Strain
    E_dyn = FloatField(null=True)     # Dynamic modulus
    pha = FloatField(null=True)       # Phase angle

# class EdynSampleSummarized(BaseModel):
#     sample_raw = ForeignKeyField(EdynSampleProcessed, backref='EdynSampleRaw', null=True)
#     sample_name = CharField()
#     collection_date = DateField(null=True)
#     E_dyn = FloatField(null=True)

    
if __name__ == "__main__":

    pass

