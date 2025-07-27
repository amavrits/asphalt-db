# -*- coding: utf-8 -*-

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from src.parsing.stiffness_parsing import read_raw_stiffness
from src.parsing.stiffness_parsing import read_summary_stiffness

def make_table_raw_data(filename, sheetname):
    dataframes_per_sheet = {}
    
    for sheet in sheetname:
        raw_data = read_raw_stiffness (filename, sheet)

        df = pd.DataFrame({
            'f': raw_data['f'],
            'eps': raw_data['eps'],
            'E_dyn': raw_data['E_dyn'],
            'pha': raw_data['pha'],   
        })
        
        dataframes_per_sheet[sheet] = df
 
    return dataframes_per_sheet

def make_table_summary_data (filename, sheetname): 
    sheet_lijst = []
    E_dyn_lijst = []

    for sheet in sheetname:
        E_dyn = read_summary_stiffness(filename, sheet)

        sheet_lijst.append(sheet)
        E_dyn_lijst.append(E_dyn)

    resultaten_df = pd.DataFrame({
        'Proefstuk': sheet_lijst,
        'E_dyn': E_dyn_lijst,
    })

    tabel_samenvatting_data = resultaten_df.sort_values(by='Proefstuk', ascending=True)
    print(tabel_samenvatting_data)
    return  


if __name__ == "__main__":
    pass