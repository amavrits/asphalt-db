# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 13:17:04 2025

@author: inge.brijker
"""

import pandas as pd

def compile_summary(filepath="ruwe_data_clean.csv"):
    df = pd.read_csv(filepath)

    # Filter op 5°C en 10 Hz
    subset = df[(df["Temperatuur (°C)"] == 5) & (df["Frequentie (Hz)"] == 10)]

    # Bereken gemiddelde E-dyn per boorkern
    summary = subset.groupby("Boorkern", as_index=False)["E-dyn (MPa)"].mean()
    summary.rename(columns={"E-dyn (MPa)": "Gemiddelde E-dyn (5°C, 10Hz)"}, inplace=True)

    return df, summary

if __name__ == "__main__":
    ruwe_data, samenvatting = compile_summary()

    ruwe_data.to_csv("output_ruwe_data.csv", index=False)
    samenvatting.to_csv("output_samenvatting.csv", index=False)

    print("Compilatie voltooid. Bestanden:")
    print("- output_ruwe_data.csv")
    print("- output_samenvatting.csv")