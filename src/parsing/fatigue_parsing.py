# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#from twdm import tqdm


def read_fatigue(filename, sheetname):
    vermoeiing = pd.read_excel(filename, sheet_name=sheetname)
    fasehoek = pd.to_numeric(vermoeiing.iloc[17, 9], errors='coerce') 
    Spanning_C = pd.to_numeric(vermoeiing.iloc[5, 13], errors='coerce') 
    Spanning_P = pd.to_numeric(vermoeiing.iloc[5, 14], errors='coerce') 
    Stijfheid = pd.to_numeric(vermoeiing.iloc[5, 15], errors='coerce')
    Nfat = pd.to_numeric(vermoeiing.iloc[5, 17], errors='coerce')
    print(f'{sheetname}: Fase Hoek = {fasehoek} [°]')
    return fasehoek, Spanning_C, Spanning_P, Stijfheid, Nfat

if __name__ == "__main__":
    
    pass
    
#we willen hier ook de rest van het aflezen in staan, de locatie van waar de data nu in o9pgeslagen is, hoe we de fatigue/strength pakken