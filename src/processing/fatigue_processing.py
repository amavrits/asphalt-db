# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#from twdm import tqdm


def calc_permanent_strain (filename, sheetname):
    
    # TODO
    # change to main directory
    
    # Pak de één na laatste waarde
    df = pd.read_excel(filename, sheet_name=sheetname)
    kolom = df.iloc[:,5]
    kolom_numeric = pd.to_numeric(kolom, errors='coerce').dropna()
    if len(kolom_numeric) >= 2:
        permanente_rek = kolom_numeric.iloc[-2]
        print(f'{sheetname}: Permanente Rek = {permanente_rek} [mm/m]')
    else:
        print(f'{sheetname}: Niet genoeg data voor Permanente Rek.')
    return permanente_rek

if __name__ == "__main__":
    
    pass