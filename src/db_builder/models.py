from peewee import *
from playhouse.postgres_ext import PostgresqlExtDatabase
from src.config import DB_CONFIG

# Initialize DB instance
db = PostgresqlExtDatabase(**DB_CONFIG)

class BaseModel(Model):
    class Meta:
        database = db

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

class Borehole(BaseModel):
    borehole_name = CharField()
    project_dike = ForeignKeyField(ProjectDike, backref="boreholes")
    collection_date = DateField(null=True)
    location = CharField(null=True)
    notes = TextField(null=True)

class StrSampleRaw(BaseModel):
    borehole = ForeignKeyField(Borehole, backref='str_samples', null=True)
    sample_name = CharField()
    collection_date = DateField(null=True)
    depth = FloatField(null=True)
    notes = TextField(null=True)

    # STR raw test fields
    t = FloatField(null=True)        # time or thickness (depends on context)
    F = FloatField(null=True)        # force
    V_org = FloatField(null=True)    # original volume

class StrSampleProcessed(BaseModel):
    sample_raw = ForeignKeyField(StrSampleRaw, backref='processed_samples', null=True)
    sample_name = CharField()
    collection_date = DateField(null=True)
    depth = FloatField(null=True)
    notes = TextField(null=True)

    # STR processed test fields
    F = FloatField(null=True)         # force
    V_cor = FloatField(null=True)     # corrected volume
    eps = FloatField(null=True)       # strain
    sig = FloatField(null=True)       # stress
    Sec = FloatField(null=True)       # secant modulus

class StrSampleSummarized(BaseModel):
    sample_processed = ForeignKeyField(StrSampleProcessed, backref='summarized_samples', null=True)
    sample_name = CharField()
    collection_date = DateField(null=True)
    depth = FloatField(null=True)
    notes = TextField(null=True)

    # STR summarized test fields
    HR = FloatField(null=True)                 # Hardening ratio?
    v = FloatField(null=True)                  # Poisson's ratio?
    sig_b = FloatField(null=True)              # Breaking stress
    eps_b = FloatField(null=True)              # Breaking strain
    Sec_10 = FloatField(null=True)             # Secant modulus at 10%
    Sec_50 = FloatField(null=True)             # Secant modulus at 50%
    Sec_100 = FloatField(null=True)            # Secant modulus at 100%
    G_c = FloatField(null=True)                # Fracture energy?
    G_c_over_eps_b = FloatField(null=True)     # G_c / eps_b
    G_c_over_eps_b_sig_b = FloatField(null=True)  # G_c / (eps_b * sig_b)
    V_Ber = FloatField(null=True)              # Some volume metric?

class FtgSampleRaw(BaseModel):
    borehole = ForeignKeyField(Borehole, backref='ftg_samples', null=True)
    sample_name = CharField()
    collection_date = DateField(null=True)
    depth = FloatField(null=True)
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
    sample_raw = ForeignKeyField(FtgSampleRaw, backref='processed_samples', null=True)
    sample_name = CharField()
    collection_date = DateField(null=True)
    depth = FloatField(null=True)
    notes = TextField(null=True)

    # FTG processed test fields
    N = IntegerField(null=True)
    eps_cycl = FloatField(null=True)   # cyclic strain
    eps_perm = FloatField(null=True)   # permanent strain
    sig_cyc = FloatField(null=True)    # cyclic stress
    sig_perm = FloatField(null=True)   # permanent stress
    E_dyn = FloatField(null=True)      # dynamic modulus
    pha = FloatField(null=True)        # phase angle

class FtgSampleSummarized(BaseModel):
    sample_processed = ForeignKeyField(FtgSampleProcessed, backref='summarized_samples', null=True)
    sample_name = CharField()
    collection_date = DateField(null=True)
    depth = FloatField(null=True)
    notes = TextField(null=True)

    # FTG summarized test fields
    sig_cyc = FloatField(null=True)    # cyclic stress
    sig_perm = FloatField(null=True)   # permanent stress
    E_ini = FloatField(null=True)      # initial modulus
    pha_ini = FloatField(null=True)    # initial phase angle
    E_50 = FloatField(null=True)       # modulus at 50% fatigue life
    pha_50 = FloatField(null=True)     # phase angle at 50% fatigue life
    N_fat = IntegerField(null=True)    # fatigue life (cycle count)

class StfSampleRaw(BaseModel):
    borehole = ForeignKeyField(Borehole, backref='samples', null=True)
    sample_name = CharField()
    collection_date = DateField(null=True)
    depth = FloatField(null=True)
    notes = TextField(null=True)

class StfSampleProcessed(BaseModel):
    sample_raw = ForeignKeyField(StfSampleRaw, backref='processed_samples', null=True)
    sample_name = CharField()
    collection_date = DateField(null=True)
    depth = FloatField(null=True)
    notes = TextField(null=True)

class StfSampleSummarized(BaseModel):
    sample_raw = ForeignKeyField(StfSampleProcessed, backref='summarized_samples', null=True)
    sample_name = CharField()
    collection_date = DateField(null=True)
    depth = FloatField(null=True)
    notes = TextField(null=True)
    
    
if __name__ == "__main__":

    pass

