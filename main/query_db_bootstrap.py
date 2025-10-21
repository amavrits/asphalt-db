import pandas as pd
import sqlite3
import os
from pathlib import Path
import seaborn as sns


if __name__ == "__main__":

    script_path = Path(__file__).parent
    db_path = script_path.parent /"data/database_all_v3.sqlite"

    conn = sqlite3.connect(db_path)

    query = """
        SELECT
        b.id AS borehole_id,
        b.project_dijk_id AS project_dike_id,
        b.aanlegjaar AS construction_year,
        b.onderzoeksjaar AS investigation_year,
        pd.project_id AS project_id,
        pd.dijk_id AS dike_id,
        p.project_name AS project_name,
        d.dike_name AS dike_name,
        d.waterboard AS waterboard,
        
        s.id AS sample_id,
        t.test_name AS test_name,
        bzs. *
    FROM Borehole b
    JOIN ProjectDijk pd ON b.project_dijk_id = pd.id
    JOIN Project p ON pd.project_id = p.id
    JOIN DIJK d ON pd.dijk_id = d.id
    JOIN Sample s ON s.borehole_id = b.id
    JOIN Test t ON t.sample_id = s.id
    JOIN BezwijksterkteSummary bzs ON bzs.test_id = t.id
    """

    df = pd.read_sql(query, conn)
    df.to_csv(db_path.parent.absolute()/"db_querry.csv", index=False)

