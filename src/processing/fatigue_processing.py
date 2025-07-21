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
        sig_perm = kolom_numeric.iloc[-2]
        print(f'{sheetname}: sig_perm = {sig_perm}')
    else:
        print(f'{sheetname}: Niet genoeg data voor Permanente Rek.')
    return sig_perm

if __name__ == "__main__":
    
    pass

