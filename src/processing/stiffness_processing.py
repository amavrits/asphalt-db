# -*- coding: utf-8 -*-

import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from src.parsing.stiffness_parsing import read_raw_stiffness

def make_table_raw_data(filename, sheetname):
    dataframes_per_sheet = {}
    
    for sheet in sheetname:
        raw_data, temp = read_raw_stiffness (filename, sheet)
        
        heading = f"Temperature [{temp:.1f} °C]"
        df = pd.DataFrame({
            'f': raw_data['f'],
            'eps': raw_data['eps'],
            'E_dyn': raw_data['E_dyn'],
            'pha': raw_data['pha'],   
        })
       
        print(f"\n{heading} - Sheet: {sheet}")
        print(df.to_string(index=False))
        
        dataframes_per_sheet[sheet] = df
    return dataframes_per_sheet

def make_table_summary_data(filename, sheetnames): 
    sheet_lijst = []
    E_dyn_lijst = []
    temp_lijst = []

    for sheet in sheetnames:
        ruwe_data, temp = read_raw_stiffness(filename, sheet)

        # Filter rijen waar f == 10
        f10_data = ruwe_data[ruwe_data['f'] == 10]

        if not f10_data.empty:
            if len(f10_data) == 1:
                E_dyn_value = f10_data['E_dyn'].iloc[0]
            else:
                E_dyn_value = f10_data['E_dyn'].mean()
        else:
            E_dyn_value = np.nan

        sheet_lijst.append(sheet)
        E_dyn_lijst.append(E_dyn_value)
        temp_lijst.append(temp)

    resultaten_df = pd.DataFrame({
        'Proefstuk': sheet_lijst,
        'E_dyn_summary (f=10)': E_dyn_lijst,
        'Temp (°C)': temp_lijst
    })

    print("\nSamenvattingstabel (gesorteerd op Proefstuk):")
    tabel_samenvatting_data = resultaten_df.sort_values(by='Proefstuk', ascending=True)
    print(tabel_samenvatting_data.to_string(index=False))

    return tabel_samenvatting_data


if __name__ == "__main__":
    pass
