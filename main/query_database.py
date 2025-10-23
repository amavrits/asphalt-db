import pandas as pd
import sqlite3
import os
from pathlib import Path
import seaborn as sns


def query_dikes_and_projects(conn: sqlite3.Connection) -> pd.DataFrame:
    #connect to database using sqlite3/pands


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
        d.waterboard as waterboard
    FROM Borehole b
    JOIN ProjectDijk pd ON b.project_dijk_id = pd.id
    JOIN Project p ON pd.project_id = p.id
    JOIN DIJK d ON pd.dijk_id = d.id
    """


    # Read into a pandas DataFrame
    dikes_and_projects = pd.read_sql(query, conn)
    dikes_and_projects['age_at_investigation'] = dikes_and_projects['investigation_year'] - dikes_and_projects['construction_year']

    return dikes_and_projects

def query_bezwijksterkte_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    # #load the BezwijksterkteSummary table (all columns) and connect to Test, Sample and Borehole to get the corresponding Borehole

    query = """
    SELECT 
        b.id AS borehole_id,
        s.id AS sample_id,
        t.test_name AS test_name,
        bzs.*
    FROM Borehole b
    JOIN Sample s ON s.borehole_id = b.id
    JOIN Test t ON t.sample_id = s.id
    JOIN BezwijksterkteSummary bzs ON bzs.test_id = t.id
    """
    df_bzs = pd.read_sql(query, conn)

    #drop HR column because it has only 0's
    # df_bzs = df_bzs.drop(columns=['HR'])
    df_bzs = df_bzs.drop_duplicates(subset=['sample_id', 'test_name'], keep='first')

    df_bzs.sort_values('borehole_id').reset_index(drop=True, inplace=True)
    return df_bzs

def query_fatigue_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    # #load the VermoeiingSummary table (all columns)

    query = """
    SELECT
        b.id AS borehole_id,
        s.id AS sample_id,
        t.test_name AS test_name,
        vs.*
    FROM Borehole b
    JOIN Sample s ON s.borehole_id = b.id
    JOIN Test t ON t.sample_id = s.id
    JOIN VermoeiingSummary vs ON vs.test_id = t.id

    """

    df_vs = pd.read_sql(query, conn)
    #drop duplicate rows based on sample_id and test_name and ensure that dropped duplicates have the same values in all columns
    df_vs = df_vs.drop_duplicates(subset=['sample_id', 'test_name'], keep='first')
    df_vs.drop(columns=['id'], inplace=True)
    return df_vs

def query_get_general_data(conn: sqlite3.Connection) -> pd.DataFrame:
    #get HR & bitumen

    query = """
    SELECT
        b.id AS borehole_id,
        g.*
    FROM GeneralData g
    JOIN Sample s ON s.id = g.sample_id
    JOIN Borehole b ON b.id = s.borehole_id
    """

    df_general = pd.read_sql(query, conn)

    #check if all HR and bitumen are the same for each borehole_id
    df_general = df_general[['borehole_id', 'HR', 'bitumen']].drop_duplicates().sort_values('borehole_id')
    # only 1 entry per borehole_id

    if df_general['borehole_id'].value_counts().max() > 1:
        raise Exception("Warning: There are different HR and/or bitumen values for some borehole_id in df_general. Please check the GeneralData.")
    return df_general

def load_and_merge_data(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    dikes_and_projects = query_dikes_and_projects(conn)
    df_bzs = query_bezwijksterkte_summary(conn)
    df_vs = query_fatigue_summary(conn)
    df_general = query_get_general_data(conn)

    #join general data to dikes_and_projects on borehole_id
    df_merge_1 = pd.merge(dikes_and_projects, df_general, on='borehole_id', how = 'left')

    #join df_bzs to dikes_and_projects on borehole_id
    df_merge_2 = pd.merge(df_merge_1, df_bzs, on='borehole_id', how='left',suffixes=('_dike', '_bzs'))
    df_merge_2.head()

    #join df_vs to df_merge_1 on borehole_id
    df_full = pd.merge(df_merge_2, df_vs, on='borehole_id', how='left', suffixes=('_bzs', '_vs'))

    conn.close()
    return df_full

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

