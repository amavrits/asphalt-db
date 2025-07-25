# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from src.parsing.fatigue_parsing import read_raw_fatigue
from src.parsing.fatigue_parsing import read_summary_fatigue
from src.parsing.fatigue_parsing import read_processed_fatigue

def make_table_raw_data(filename, sheetname):
    dataframes_per_sheet = {}
    
    for sheet in sheetname:
        raw_data = read_raw_fatigue (filename, sheet)

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
 
    return dataframes_per_sheet

def make_table_processed_data(filename, sheetname):
    dataframes_per_sheet = {}

    for sheet in sheetname:
        processed_data = read_processed_fatigue (filename, sheet)
                                              
        df = pd.DataFrame({
            'N': processed_data['N'],
            'eps_cycl': processed_data['eps_cycl'],
            'sig_cyc': processed_data['sig_cyc'],
            'sig_perm': processed_data['sig_perm'],
            'E_dyn': processed_data['E_dyn'],
            'pha': processed_data['pha'],
        })
        
        dataframes_per_sheet[sheet] = df

    return dataframes_per_sheet

def make_table_summary_data (filename, sheetname): 
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
        'Proefstuk': sheet_lijst,
        'pha_ini': pha_ini_lijst,
        'pha_50': pha_50_lijst,
        'sig_cyc': sig_cyc_lijst,
        'sig_perm': sig_perm_lijst,
        'E_ini': E_ini_lijst,
        'E_50': E_50_lijst,
        'N_fat': N_fat_lijst
    })

    tabel_samenvatting_data = resultaten_df.sort_values(by='Proefstuk', ascending=True)
    print(tabel_samenvatting_data)
    return  

if __name__ == "__main__":
    pass
