# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 13:55:27 2025

@author: marloes.slokker
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def make_headtable(file_path, projectnummers, dijktraject):
    data = pd.read_excel(file_path)
    data['Projectnummer'] = data['Projectnummer'].astype(str)
    data['Aanlegjaar'] = pd.to_numeric(data['Aanleg-jaar'], errors='coerce')
    data['Onderzoeksjaar'] = pd.to_numeric(data['onderzoeksjaar'], errors='coerce')
    data['Dijktraject'] = dijktraject
    data['LFT'] = data['Onderzoeksjaar'] - data['Aanlegjaar']
    data = data[['Projectnummer', 'Dijknaam', 'Kilometrering dijkvak', 'Waterschap / beheerder',
              'Dijktraject', 'Aanlegjaar', 'Onderzoeksjaar', 'LFT']].copy()
    data.rename(columns={'Waterschap / beheerder': 'Dijkbeheerder'}, inplace=True)
    
    for nummer in projectnummers:
        df = data[data['Projectnummer'] == nummer].copy()
    
        if df.empty:
            print(f"[!] Projectnummer {nummer} not found.")
            continue
    
        # Drop fully duplicate rows
        df_merged = df.drop_duplicates()
    
        # Output
        print(f"Projectnummer: {nummer}")
        print(df_merged)
        
    return df_merged