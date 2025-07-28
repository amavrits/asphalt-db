
# read from the database
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
import psycopg
from src.db_builder.models import *

from src.config import DB_CONFIG



project = "P_1"


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



query_strength_data =  (
    StrSummary
    .select()
    .join(Test)
    .join(Sample)
    .join(Borehole)
    .join(ProjectDike)
    .join(Project)
    .where(
        (Project.project_name == project)
    )
)

query_fatigue_data =  (
    FtgSummary
    .select()
    .join(Test)
    .join(Sample)
    .join(Borehole)
    .join(ProjectDike)
    .join(Project)
    .where(
        (Project.project_name == project)
    )
)


data_str = [
    {
        "sig_b": row.sig_b,  #
    }
    for row in query_strength_data
]

data_ftg = [
    {
        "pha_ini": row.pha_ini,  #
    }
    for row in query_fatigue_data
]


# Make sure the pairs are from the same borehole: data = {'BH1': {'sig_b': 10, 'pha_ini': 0.5}, ...}
print(data_str)
print(data_ftg)

# plot
plt.figure(figsize=(10, 6))
borehole = [1,2,3,4]

plt.plot(data_str[0]["sig_b"], data_ftg[0]["pha_ini"], 'o', label=f'Borehole {borehole[0]}')
plt.plot(data_str[1]["sig_b"], data_ftg[1]["pha_ini"], 'o', label=f'Borehole {borehole[1]}')
plt.plot(data_str[2]["sig_b"], data_ftg[2]["pha_ini"], 'o', label=f'Borehole {borehole[2]}')
plt.plot(data_str[3]["sig_b"], data_ftg[3]["pha_ini"], 'o', label=f'Borehole {borehole[3]}')

plt.title(f"Strength vs Fatigue for project 1900384")
plt.xlabel("Strength (sig_b)")
plt.ylabel("Fatigue (pha_ini)")
plt.grid()
plt.legend()
plt.tight_layout()
# plt.savefig(f"strength_vs_fatigue_{project}.png")
plt.show()