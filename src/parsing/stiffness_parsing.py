# -*- coding: utf-8 -*-

import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def read_raw_stiffness (filename, sheet):
    ruwe_data = pd.read_excel(filename, sheet_name=sheet)
    temp = ruwe_data.iloc[2,7]
    
    ruwe_data = pd.read_excel(filename, sheet_name=sheet, skiprows=18)

    ruwe_data['f']= ruwe_data.iloc[:,1]
    ruwe_data['eps']= ruwe_data.iloc[:,2]
    ruwe_data['E_dyn']= ruwe_data.iloc[:,3]
    ruwe_data['pha']= ruwe_data.iloc[:,4]
    ruwe_data = ruwe_data.apply(pd.to_numeric, errors='coerce')  

    ruwe_data = ruwe_data [['f', 'eps', 'E_dyn','pha']].copy()
    ruwe_data = ruwe_data.dropna()
    return ruwe_data, temp


if __name__ == "__main__":
    pass

