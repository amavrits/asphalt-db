# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#from twdm import tqdm
from src.parsing.fatigue_parsing import read_raw_fatigue
from src.parsing.fatigue_parsing import read_summary_fatigue
from src.parsing.fatigue_parsing import read_processed_fatigue


def make_table_summary (filename, sheetname):  # let op: sheetname is hier een lijst
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

def make_table_processed_data(file_path, grafiektitel, sheet):
    f = pd.ExcelFile(file_path)
    alle_sheets = f.sheet_names
    sheetnames = alle_sheets[3:]  # vanaf sheet index 3
    
     

    # Dictionary om DataFrames per sheetnaam op te slaan
    dataframes_per_sheet = {}

    for sheet in sheetnames:
        originele_data, raw_data = read_data(file_path, sheet)
        D, h, strength, v = read_parameters(file_path, sheet)
        xmean = originele_data['Verplaatsing'].rolling(8).mean()
        ymean = originele_data['Kracht'].rolling(8).mean()
        max_index = ymean.idxmax()
        final_line, rc, intercept, _ = calc_linear_fit(xmean, ymean, max_index)

        gecorrigeerde_data = originele_data.copy()
        gecorrigeerde_data = correct_data(gecorrigeerde_data, rc, intercept)
        verplaatsing_corr = gecorrigeerde_data['Verplaatsing']
        process_data = define_sec_modulus(file_path, sheet, gecorrigeerde_data, D, h)[3]

        # Maak een dataframe voor deze sheet
        df = pd.DataFrame({
            'F': gecorrigeerde_data['Kracht'],
            # 'V_org': raw_data['verplaatsing'],
            'V_cor': verplaatsing_corr,
            'eps': process_data['rek'],
            'sig': process_data['spanning'],
            'Sec': process_data['secantmodulus']
        })

        # Voeg toe aan dictionary met sheetnaam als key
        dataframes_per_sheet[sheet] = df

    # Voorbeeld: print eerste paar rijen van elke sheet
    for naam, df in dataframes_per_sheet.items():
        print(f"Proefstuk: {naam}")
        print(df.head(), "\n")

    return dataframes_per_sheet