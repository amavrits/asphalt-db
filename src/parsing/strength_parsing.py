# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 16:42:49 2025

@author: marloes.slokker
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#from twdm import tqdm

def read_data(file_path, sheet):
    
    data = pd.read_excel(file_path, sheet_name=sheet, skiprows=16)
    data = data[['Verplaatsing', 'Kracht']].copy()
    data['Verplaatsing'] = pd.to_numeric(data['Verplaatsing'], errors='coerce')
    data['Kracht'] = pd.to_numeric(data['Kracht'], errors='coerce')
    data = data.dropna()
            
    raw_data = pd.read_excel(file_path, sheet_name=sheet, skiprows=17)
    raw_data['tijd'] = raw_data.iloc[:, 27]
    raw_data['verplaatsing'] = raw_data.iloc[:, 26]
    raw_data['kracht'] = raw_data.iloc[:, 25]
    raw_data = raw_data[['tijd', 'kracht', 'verplaatsing']].copy()
    
    raw_data = raw_data.dropna()
    # print(raw_data)
    return data, raw_data

def read_parameters(file_path, sheet):
    values = pd.read_excel(file_path, sheet_name=sheet)
    D = pd.to_numeric(values.iloc[4, 2], errors='coerce') / 1000
    h = pd.to_numeric(values.iloc[5, 2], errors='coerce') / 1000
    # L = pd.to_numeric(values.iloc[9, 2], errors='coerce') / 1000
    strength = pd.to_numeric(values.iloc[7, 7], errors='coerce')
    v = pd.to_numeric(values.iloc[10, 7], errors='coerce')
    
    return D, h, strength, v

    