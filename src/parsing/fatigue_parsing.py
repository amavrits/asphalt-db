# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#from twdm import tqdm

filename = 'Vermoeiing vak 1 (1-8) Versie2.xlsm'
f = pd.ExcelFile(filename)

alle_sheets = f.sheet_names
sheetname = alle_sheets[3:]

def read_raw_fatigue(filename, sheet):
    # Lees sheet in vanaf rij 14 (skiprows=13), zodat index 0 == Excel rij 14
    ruwe_data = pd.read_excel(filename, sheet_name=sheet, skiprows=14)
    
    # Pak kolommen AE t/m AK (kolomindex 30 t/m 36), vanaf tweede rij (index 1 = Excel rij 15)
    ruwe_data['MaximumStroke']= ruwe_data.iloc[:,30]
    ruwe_data['MinimumStroke']= ruwe_data.iloc[:,31]
    ruwe_data['PeakToPeakStroke']= ruwe_data.iloc[:,32]
    ruwe_data['MaximumLoad']= ruwe_data.iloc[:,33]
    ruwe_data['PeakToPeakLoad']= ruwe_data.iloc[:,34]
    ruwe_data['InPhaseModulus']= ruwe_data.iloc[:,35]
    ruwe_data['OutPhaseModulus']= ruwe_data.iloc[:,36]
    ruwe_data = ruwe_data.apply(pd.to_numeric, errors='coerce')  # zet alles om naar getallen

    ruwe_data = ruwe_data [['MaximumStroke', 'MinimumStroke', 'PeakToPeakStroke',
                            'MaximumLoad', 'PeakToPeakLoad', 'InPhaseModulus', 'OutPhaseModulus']].copy()
    ruwe_data = ruwe_data.dropna()
    print (ruwe_data)
    # ruwe_data['sheetname'] = sheetname

    return ruwe_data




def read_summary_fatigue (filename, sheetname): #dit is summary data
    vermoeiing = pd.read_excel(filename, sheet_name=sheetname)
    pha_ini = pd.to_numeric(vermoeiing.iloc[17, 9], errors='coerce')
    
    kolom1 = pd.to_numeric(vermoeiing.iloc[13:, 1], errors='coerce').dropna().reset_index(drop=True)           # Kolom 1 vanaf rij 14
    doelwaarde = 0.5 * kolom1.max()
    index_bij_50 = (kolom1 - doelwaarde).abs().idxmin()
    
    pha_50 = pd.to_numeric(vermoeiing.iloc[index_bij_50, 9], errors='coerce') #fix calculation 50% lastherhalingen
    sig_cyc = pd.to_numeric(vermoeiing.iloc[5, 13], errors='coerce') 
    sig_perm = pd.to_numeric(vermoeiing.iloc[5, 14], errors='coerce') 
    E_ini = pd.to_numeric(vermoeiing.iloc[17, 8], errors='coerce') # na 50 lastherhalingen
    E_50 = pd.to_numeric(vermoeiing.iloc[index_bij_50, 8], errors='coerce') #fix calculation 50% lastherhalingen
    N_fat = pd.to_numeric(vermoeiing.iloc[5, 17], errors='coerce')
    print (pha_ini, pha_50, sig_cyc, sig_perm, E_ini, E_50, N_fat)
    return pha_ini, pha_50, sig_cyc, sig_perm, E_ini, E_50, N_fat


for i, sheet in enumerate(sheetname):
    read_summary_fatigue(filename, sheet)


if __name__ == "__main__":
    
    pass
    
#we willen hier ook de rest van het aflezen in staan, de locatie van waar de data nu in o9pgeslagen is, hoe we de fatigue/strength pakken

