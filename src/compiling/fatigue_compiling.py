# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#from twdm import tqdm
from sqlalchemy import create_engine
from sqlalchemy.types import Float, String, Integer

from src.parsing.fatigue_parsing import read_raw_fatigue
from src.parsing.fatigue_parsing import read_summary_fatigue
from src.processing.fatigue_processing import calc_permanent_strain

# TODO
#db_url = "postgresql://username:password@localhost:5432/your_database"
filename = 'Vermoeiing vak 1 (1-8) Versie2.xlsm'
f = pd.ExcelFile(filename)

alle_sheets = f.sheet_names
sheetname = alle_sheets[3:] #vanaf sheet index 4

def data_toevoegen_samenvatting(filename, sheetname):  # let op: sheetname is hier een lijst
    sheet_lijst = []
    pha_ini_lijst = []
    pha_50_lijst = []
    sig_cyc_lijst = []
    sig_perm_lijst = []
    E_ini_lijst = []
    E_50_lijst = []
    N_fat_lijst = []

    for sheet in sheetname:
        pha_ini, pha_50, sig_cyc, sig_perm, E_ini, E_50, N_fat = read_summary_fatigue(filename, sheet)

        sheet_lijst.append(sheet)
        pha_ini_lijst.append(pha_ini)
        pha_50_lijst.append(pha_50)
        sig_cyc_lijst.append(sig_cyc)
        sig_perm_lijst.append(sig_perm)
        E_ini_lijst.append(E_ini)
        E_50_lijst.append(E_50)
        N_fat_lijst.append(N_fat)
        

    resultaten_df = pd.DataFrame({
        'sheetname': sheet_lijst,
        'pha_ini': pha_ini_lijst,
        'pha_50': pha_50_lijst,
        'sig_cyc': sig_cyc_lijst,
        'sig_perm': sig_perm_lijst,
        'E_ini': E_ini_lijst,
        'E_50': E_50_lijst,
        'N_fat': N_fat_lijst
    })

    tabel_samenvatting_data = resultaten_df.sort_values(by='sheetname', ascending=True)
    # print(tabel_samenvatting_data)
    return tabel_samenvatting_data 

def make_table_raw_data(filename, sheetname):
 
    dataframes_per_sheet = {}
    
    for sheet in sheetname:
        raw_data = read_raw_fatigue (filename, sheetname)
 
        # Maak een dataframe voor deze sheet
        df = pd.DataFrame({
            'MaximumStroke': raw_data['MaximumStroke'],
            'MinimumStroke': raw_data['MinimumStroke'],
            'PeakToPeakStroke': raw_data['PeakToPeakStroke'],
            'MaximumLoad': raw_data['MaximumLoad'],
            'PeakToPeakLoad': raw_data['PeakToPeakLoad'],
            'InPhaseModulus': raw_data['InPhaseModulus'],
            'OutPhaseModulus': raw_data['OutPhaseModulus'],   
        })
        
        dataframes_per_sheet[sheet] = df
 
    # Voorbeeld: print eerste paar rijen van elke sheet
    for naam, df in dataframes_per_sheet.items():
        print(f"Sheet: {naam}")
        print(df.head(), "\n")
 
    return dataframes_per_sheet
    
   
# TODO 
# function die ruwe data roept
# function die berekende data roept
# function die summary data roept 

if __name__ == "__main__":
    pass


for i, sheet in enumerate(sheetname):
    read_raw_fatigue(filename, sheet)
    read_summary_fatigue(filename, sheet)
    
read_summary_fatigue(filename, sheet)

# ruwe_data = make_table_raw_data(filename, sheetname)
# print(f"Ruwe data uit sheet {sheetname}:\n", ruwe_data.head())

# summary = read_summary_fatigue(filename, sheetname)
# print(f"Samenvatting uit sheet {sheetname}:\n", summary)