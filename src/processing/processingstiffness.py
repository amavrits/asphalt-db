# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 13:16:51 2025

@author: inge.brijker
"""

import pandas as pd

def process_data(filepath="ruwe_data_10hz.csv"):
    df = pd.read_csv(filepath)

    # Forceer juiste types
    df["Boorkern"] = df["Boorkern"].astype(str)
    df["Temperatuur (°C)"] = pd.to_numeric(df["Temperatuur (°C)"], errors="coerce")
    df["Frequentie (Hz)"] = pd.to_numeric(df["Frequentie (Hz)"], errors="coerce")
    df["Rek bij breuk"] = pd.to_numeric(df["Rek bij breuk"], errors="coerce")
    df["E-dyn (MPa)"] = pd.to_numeric(df["E-dyn (MPa)"], errors="coerce")
    df["Phasehoek (°)"] = pd.to_numeric(df["Phasehoek (°)"], errors="coerce")

    # Filter alleen geldige E-dyn en Temperatuur
    df_clean = df.dropna(subset=["E-dyn (MPa)", "Temperatuur (°C)"])

    return df_clean

if __name__ == "__main__":
    df_clean = process_data()
    df_clean.to_csv("ruwe_data_clean.csv", index=False)
    print("Verwerking afgerond. Schone data opgeslagen in 'ruwe_data_clean.csv'")