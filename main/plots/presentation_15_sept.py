from pathlib import Path

import matplotlib.pyplot as plt

import pandas as pd

from src.db_builder import models
from src.db_builder.models import *

from src.db_builder.models import VermoeiingSummary
import matplotlib.pyplot as plt
from peewee import fn

# db_config = DB_CONFIG
# db_name = db_config["database"]
# admin_conn = psycopg.connect(
#         dbname="postgres",
#         user=db_config["user"],
#         password=db_config["password"],
#         host=db_config["host"],
#         port=db_config["port"],
#         autocommit=True
#     )
#
# db.connect()


DB_PATHH = Path(r"c:\Users\hauth\OneDrive - Stichting Deltares\projects\Asphalte Regression\DB").joinpath(
    "asfalt_db_v1.sqlite3")
# from peewee import SqliteDatabase
db = SqliteDatabase(DB_PATHH)
db.connect()

# Optional: Bind models if needed
db.bind([models.Project, models.Borehole, models.Sample, models.Test])

# Example read
# for project in models.Project.select():
#     print(project.project_name, project.project_code)

db.close()


def plot_age_strength_hue_vormfactor():
    query = (
        BezwijksterkteSummary
        .select(
            BezwijksterkteSummary.sig_b,
            BezwijksterkteSummary.V_Ber,  # <- extra field for color
            Sample,
            Borehole
        )
        .join(Sample, on=(BezwijksterkteSummary.sample_name == Sample.sample_name))
        .join(Borehole, on=(Sample.borehole == Borehole.id))
        .where(
            (BezwijksterkteSummary.sig_b.is_null(False)) &
            (Borehole.aanlegjaar.is_null(False)) &
            (Borehole.onderzoeksjaar.is_null(False)) &
            (BezwijksterkteSummary.v.is_null(False))
        )
    )

    x, y, c = [], [], []

    for row in query:
        borehole = row.sample.borehole
        leeftijd = borehole.onderzoeksjaar - borehole.aanlegjaar
        if leeftijd >= 0:
            x.append(leeftijd)
            y.append(row.sig_b)
            c.append(row.V_Ber)

    # Scatter plot with color mapping
    plt.figure(figsize=(8, 6))
    print(len(x))
    sc = plt.scatter(x, y, c=c, cmap="viridis", alpha=0.8)
    # write in a csv file x and y and c:
    plt.xlabel("Leeftijd (jaar)")
    plt.ylabel("Sterkte bij bezwijking (sig_b)")
    plt.title("Bezwijksterkte vs Leeftijd (kleur = vormfactor)")
    plt.colorbar(sc, label="vormfactor")  # Add colorbar legend
    plt.grid(True)
    plt.show()


def plot_age_strength_hue_pha50():
    StrengthSummary = BezwijksterkteSummary.alias()
    FatigueSummary = VermoeiingSummary.alias()

    query = (
        BezwijksterkteSummary
        .select(
            BezwijksterkteSummary.sig_b,
            FatigueSummary.pha_50.alias("pha_50"),
            Borehole.aanlegjaar,
            Borehole.onderzoeksjaar
        )
        .join(Sample, on=(BezwijksterkteSummary.sample_name == Sample.sample_name))
        .join(Borehole, on=(Sample.borehole == Borehole.id))
        .switch(BezwijksterkteSummary)
        .join(FatigueSummary, on=(BezwijksterkteSummary.sample_name == FatigueSummary.sample_name))
        .where(
            (BezwijksterkteSummary.sig_b.is_null(False)) &
            (FatigueSummary.pha_50.is_null(False)) &
            (Borehole.aanlegjaar.is_null(False)) &
            (Borehole.onderzoeksjaar.is_null(False))
        )
    )

    x, y, c = [], [], []

    for row in query:
        borehole = row.sample.borehole
        leeftijd = borehole.onderzoeksjaar - borehole.aanlegjaar
        if leeftijd >= 0:
            x.append(leeftijd)
            y.append(row.sig_b)
            c.append(row.vermoeiingsummary.pha_50)

    # Scatter plot with pha_60 as color
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=c, cmap="plasma", alpha=0.8)
    plt.xlabel("Leeftijd (jaar)")
    plt.ylabel("Sterkte bij bezwijking (sig_b)")
    plt.title("Bezwijksterkte vs Leeftijd")
    plt.colorbar(sc, label="pha_50")
    plt.grid(True)
    plt.show()


def plot_age_strength_hue_HR():
    StrengthSummary = BezwijksterkteSummary.alias()
    FatigueSummary = VermoeiingSummary.alias()

    query = (
        BezwijksterkteSummary
        .select(

            BezwijksterkteSummary.sig_b,
            FatigueSummary.pha_50.alias("pha_50"),
            Borehole.aanlegjaar,
            Borehole.onderzoeksjaar,
            GeneralData.HR,

        )
        .join(Sample, on=(BezwijksterkteSummary.sample_name == Sample.sample_name))
        .join(Borehole, on=(Sample.borehole == Borehole.id))
        .switch(BezwijksterkteSummary)
        .join(FatigueSummary, on=(BezwijksterkteSummary.sample_name == FatigueSummary.sample_name))
        .join(Sample, on=(GeneralData.sample == Sample.id))

        .where(
            (BezwijksterkteSummary.sig_b.is_null(False)) &
            (FatigueSummary.pha_50.is_null(False)) &
            (Borehole.aanlegjaar.is_null(False)) &
            (Borehole.onderzoeksjaar.is_null(False))
        )
    )

    x, y, c, HR = [], [], [], []

    for row in query:
        borehole = row.sample.borehole
        leeftijd = borehole.onderzoeksjaar - borehole.aanlegjaar
        if leeftijd >= 0:
            x.append(leeftijd)
            y.append(row.sig_b)
            c.append(row.vermoeiingsummary.pha_50)
            HR.append(row['HR'])

    # create csv file with x, y, c and HR
    df = pd.DataFrame({'Leeftijd': x, 'Sterkte': y, 'pha_50': c, 'HR': HR})
    df.to_csv('age_strength_pha50_HR.csv', index=False)

    # Scatter plot with pha_60 as color
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=c, cmap="plasma", alpha=0.8)
    plt.xlabel("Leeftijd (jaar)")
    plt.ylabel("Sterkte bij bezwijking (sig_b)")
    plt.title("Bezwijksterkte vs Leeftijd")
    plt.colorbar(sc, label="pha_50")
    plt.grid(True)
    plt.show()


def plot_HR_vs_pha50():
    FatigueSummary = VermoeiingSummary.alias()

    query = (
        GeneralData
        .select(
            GeneralData.HR,
            FatigueSummary.pha_50.alias("pha_50")
        )
        .join(Sample, on=(GeneralData.sample == Sample.id))
        .switch(Sample)
        .join(FatigueSummary, on=(Sample.sample_name == FatigueSummary.sample_name))
        .where(
            (GeneralData.HR.is_null(False)) &
            (FatigueSummary.pha_50.is_null(False))
        )
        .dicts()
    )

    x, y = [], []

    for row in query:
        x.append(row['HR'])
        y.append(row['pha_50'])

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.8)
    plt.xlabel("HR (void ratio)")
    plt.ylabel("pha_50 (phase angle at 50% cycles)")
    plt.title("HR vs pha_50")
    plt.grid(True)
    plt.show()


# plot_age_strength_hue_vormfactor()
# plot_age_strength_hue_pha50()
#
plot_age_strength_hue_HR()
# plot_HR_vs_pha50()
