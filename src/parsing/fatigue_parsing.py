# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def read_raw_fatigue(filename, sheet):
    ruwe_data = pd.read_excel(filename, sheet_name=sheet, skiprows=14)
    
    ruwe_data['MaximumStroke']= ruwe_data.iloc[:,30]
    ruwe_data['MinimumStroke']= ruwe_data.iloc[:,31]
    ruwe_data['PeakToPeakStroke']= ruwe_data.iloc[:,32]
    ruwe_data['MaximumLoad']= ruwe_data.iloc[:,33]
    ruwe_data['PeakToPeakLoad']= ruwe_data.iloc[:,34]
    ruwe_data['InPhaseModulus']= ruwe_data.iloc[:,35]
    ruwe_data['OutPhaseModulus']= ruwe_data.iloc[:,36]
    ruwe_data = ruwe_data.apply(pd.to_numeric, errors='coerce')  

    ruwe_data = ruwe_data [['MaximumStroke', 'MinimumStroke', 'PeakToPeakStroke',
                            'MaximumLoad', 'PeakToPeakLoad', 'InPhaseModulus', 'OutPhaseModulus']].copy()
    ruwe_data = ruwe_data.dropna()

    return ruwe_data

def read_processed_fatigue (filename, sheet):
    processed_data = pd.read_excel(filename, sheet_name=sheet, skiprows=12)
    
    processed_data['N'] = processed_data.iloc[:,2]
    processed_data['eps_cycl'] = processed_data.iloc[:,4]
    processed_data['eps_perm'] = processed_data.iloc[:,5]
    processed_data['sig_cyc'] = processed_data.iloc[:,6]
    processed_data['sig_perm'] = processed_data.iloc[:,7]
    processed_data['E_dyn'] = processed_data.iloc[:,8]
    processed_data['pha'] = processed_data.iloc[:,9]
    processed_data = processed_data.apply(pd.to_numeric, errors='coerce')

    processed_data = processed_data [['N', 'eps_cycl', 'eps_perm', 'sig_cyc','sig_perm', 'E_dyn', 'pha']].copy()
    processed_data = processed_data.dropna(how="all")

    return processed_data

def read_summary_fatigue (filename, sheet):
    vermoeiing = pd.read_excel(filename, sheet_name=sheet)
    pha_ini = pd.to_numeric(vermoeiing.iloc[17, 9], errors='coerce')
    sig_cyc = pd.to_numeric(vermoeiing.iloc[5, 13], errors='coerce') 
    sig_perm = pd.to_numeric(vermoeiing.iloc[5, 14], errors='coerce') 
    E_ini = pd.to_numeric(vermoeiing.iloc[17, 8], errors='coerce') 
    N_fat = pd.to_numeric(vermoeiing.iloc[5, 17], errors='coerce')
    
    N = pd.to_numeric(vermoeiing.iloc[13:, 2], errors='coerce').reset_index(drop=True)
    pha = pd.to_numeric(vermoeiing.iloc[13:, 9], errors='coerce').reset_index(drop=True)
    E = pd.to_numeric(vermoeiing.iloc[13:, 8], errors='coerce').reset_index(drop=True)

    eerste_nan_index = pha[pha.isna()].index.min()
    laatste_index = eerste_nan_index - 1 if not pd.isna(eerste_nan_index) else len(pha) - 1

    N_geldig = N.loc[:laatste_index]
    pha_geldig = pha.loc[:laatste_index]
    E_geldig = E.loc[:laatste_index]

    if N_geldig.empty:
        return pha_ini, float('nan'), sig_cyc, sig_perm, E_ini, float('nan'), N_fat

    doel_N = 0.5 * N_geldig.max()
    index_bij_50 = (N_geldig - doel_N).abs().idxmin()

    pha_50 = pha_geldig.loc[index_bij_50]
    E_50 = E_geldig.loc[index_bij_50]
   
    return pha_ini, pha_50, sig_cyc, sig_perm, E_ini, E_50, N_fat

if __name__ == "__main__":
    
    pass
    
