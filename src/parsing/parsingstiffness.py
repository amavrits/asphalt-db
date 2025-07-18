# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 13:16:31 2025

@author: inge.brijker
"""

import pandas as pd
import re

def parse_sheet(sheet, xls):
    df = pd.read_excel(xls, sheet_name=sheet, header=None)

    # Boorkern uit C5
    boorkern = str(df.iloc[4, 2]).strip()

    # Temperatuur uit B17
    temp_text = str(df.iloc[16, 1])
    match = re.search(r"(\d+)\s*°C", temp_text)
    temperatuur = int(match.group(1)) if match else None

    # Meetdata vanaf rij 20
    data = df.iloc[19:, :]  # Excel is 0-gebaseerd, dus rij 20 = index 19

    # Filter op frequentie 10 Hz in kolom B (index 1)
    data_10hz = data[data.iloc[:, 1] == 10]

    # Haal relevante kolommen op (check of dit klopt voor jouw bestand):
    # Frequentie: kolom 1 (B), Rek bij breuk: kolom 5 (F), E-dyn: kolom 6 (G), Phasehoek: kolom 7 (H)
    parsed = pd.DataFrame({
        "Boorkern": boorkern,
        "Temperatuur (°C)": temperatuur,
        "Frequentie (Hz)": data_10hz.iloc[:, 1],
        "Rek bij breuk": data_10hz.iloc[:, 5],
        "E-dyn (MPa)": data_10hz.iloc[:, 6],
        "Phasehoek (°)": data_10hz.iloc[:, 7],
    })

    return parsed

def parse_all_sheets(filepath):
    xls = pd.ExcelFile(filepath, engine="openpyxl")
    relevant_sheets = xls.sheet_names[4:12]  # tabbladen 5 t/m 12

    all_data = pd.concat([parse_sheet(sheet, xls) for sheet in relevant_sheets], ignore_index=True)
    return all_data

if __name__ == "__main__":
    # Pad naar Excelbestand
    bestand = "Analyse Stijfheid 3pb v3.0.xlsm"
    
    ruwe_data = parse_all_sheets(bestand)
    
    # Optioneel: sla op als CSV
    ruwe_data.to_csv("ruwe_data_10hz.csv", index=False)
    print("Parsing afgerond. Data opgeslagen in 'ruwe_data_10hz.csv'")