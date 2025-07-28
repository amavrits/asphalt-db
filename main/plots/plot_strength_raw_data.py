# read from the database
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
import psycopg
from src.db_builder.models import *

from src.config import DB_CONFIG
from src.processing.strength_processing import calc_fracture_data, calc_linear_fit, correct_data, make_plot



borehole = "BH1"
sample = "1B"


db_config = DB_CONFIG
db_name = db_config["database"]
admin_conn = psycopg.connect(
        dbname="postgres",
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"],
        autocommit=True
    )


db.connect()

## Get sample Data
query_sample_data =  (
    Sample
    .select()
    .join(Borehole)
    .where(
        (Sample.sample_name == sample) &
        (Borehole.borehole_name == borehole)
    )
)
v = query_sample_data[0].v
D = query_sample_data[0].thickness
h = query_sample_data[0].height
strength = query_sample_data[0].strength

## Get raw data
query_raw_data =  (
    StrSampleRaw
    .select()
    .join(Test)
    .join(Sample)
    .join(Borehole)
    .where(
        (Borehole.borehole_name == borehole)
    )
)

data = [
    {
        "Verplaatsing": -row.V_org,  #TODO why is it negative in db ???
        "Kracht": -row.F  #TODO why is it negative in db ???
    }
    for row in query_raw_data
]
originele_data = pd.DataFrame(data)


#HOOFDSCRIPT - TABEL EN FIGUUR
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
axs = axs.flatten()
ax = axs[0]


xmean = originele_data['Verplaatsing'].rolling(8).mean()
ymean = originele_data['Kracht'].rolling(8).mean()
max_index = ymean.idxmax()
final_line, rc, intercept, _ = calc_linear_fit(xmean, ymean, max_index)

gecorrigeerde_data = originele_data.copy()
gecorrigeerde_data = correct_data(gecorrigeerde_data, rc, intercept)
#
rek_max, x_max, y_max, x_interp, y_interp, Gc, vormfactor = calc_fracture_data(gecorrigeerde_data, D, h)
#
make_plot(ax, borehole, originele_data, xmean, ymean, final_line, gecorrigeerde_data,
            x_max, y_max, x_interp, y_interp, Gc)
plt.show()

db.close()

