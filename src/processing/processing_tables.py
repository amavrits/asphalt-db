# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 13:55:38 2025

@author: marloes.slokker
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def make_table2(file_path, projectnummers):
    data = pd.read_excel(file_path)
    data['Projectnummer'] = data['Projectnummer'].astype(str)
    data['Orientatie*'] = 'n.t.b.'
    data = data[['Projectnummer', 'Dijknaam', 'Boorkern', 'dichtheid mengsel schatting', 'dichtheid mengsel', 'dichtheid proef gemeten', 
              'HR', 'Orientatie*', 'Bitumengehalte NEN', 'Bitumengehalte obv bestek', 'Bitumengehalte gerapporteerd op of in', 'Bitumengehalte "in"', 
              'bitumengehalte bepaald op kern', 'Herkomst bitumengehalte']].copy()
    data.rename(columns={'dichtheid mengsel schatting': 'D_M_G', 'dichtheid mengsel': 'D_M_B', 'dichtheid proef gemeten': 'D_P',
                       'Bitumengehalte NEN': 'BG NEN', 'Bitumengehalte obv bestek': 'BG bestek', 'Bitumengehalte gerapporteerd op of in': 'BG op of in',
                       'Bitumengehalte "in"': 'BG in', 'bitumengehalte bepaald op kern': 'BG kern', 'Herkomst bitumengehalte': 'Herkomst BG'}, inplace=True)
    
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
