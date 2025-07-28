# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 16:43:04 2025

@author: marloes.slokker
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#from twdm import tqdm
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from src.parsing.strength_parsing import read_data, read_parameters


def calc_linear_fit(xmean, ymean, max_index):
    step_size = 14  # Grootte van elk lijnfragment

    grootste_verschil = -np.inf
    beste_start_idx = None
    beste_end_idx = None
    beste_rc = None

    for start_idx in range(7, max_index - step_size + 1, step_size):
        end_idx = start_idx + step_size - 1

        verschil = abs(ymean[end_idx] - ymean[start_idx])  # Verschil in y-waarde

        if verschil > grootste_verschil:
            grootste_verschil = verschil
            beste_start_idx = start_idx
            beste_end_idx = end_idx
            beste_rc = (ymean[end_idx] - ymean[start_idx]) / (xmean[end_idx] - xmean[start_idx])

    if beste_start_idx is not None:
        rc = beste_rc
        intercept = ymean[beste_start_idx] - rc * xmean[beste_start_idx]

        x_start = (0 + (rc * xmean[beste_start_idx] - ymean[beste_start_idx])) / rc
        x_end = (ymean[max_index] + (rc * xmean[beste_start_idx] - ymean[beste_start_idx])) / rc

        x = np.linspace(x_start, x_end, 100)
        y = rc * x - (rc * xmean[beste_start_idx] - ymean[beste_start_idx])

        final_line = (x, y)

        #print(f"Grootste y-verschil gevonden tussen index {beste_start_idx} en {beste_end_idx}")
        return final_line, rc, intercept, grootste_verschil

    #print("Geen geschikte lijn gevonden.")
    return None, None, None, grootste_verschil


def correct_data(data, rc, intercept):
    data = data.dropna()
    
    # Standaard threshold nul
    threshold = 0

    # Alleen als er een geldige lineaire fit is
    if rc is not None and intercept is not None:
        # Bepaal de lineaire fit y-waarden
        line_fit_y = rc * data['Verplaatsing'] + intercept

        # Verschil tussen krachtdata en fit
        verschil = data['Kracht'] - line_fit_y

        # Zoek eerste kruispunt
        kruispunt_indices = verschil[verschil <= 0].index

        if not kruispunt_indices.empty:
            eerste_kruising_idx = kruispunt_indices[0]
            threshold = data.loc[eerste_kruising_idx, 'Verplaatsing']
        else:
            threshold = data['Verplaatsing'].max()

    mask = data['Verplaatsing'] < threshold
    if rc is not None and intercept is not None:
        corrected_x = (data.loc[mask, 'Kracht'] - intercept) / rc
        data.loc[mask, 'Verplaatsing'] = corrected_x

    min_x = data['Verplaatsing'].min()
    data['Verplaatsing'] -= min_x

    return data


def calc_fracture_data(data, D, h):
    rek = 10**6 * (12 * h * 1000) / (2 * 200**2) * (data['Verplaatsing'] - data['Verplaatsing'].iloc[1])
    data['Rek'] = rek

    x = data['Verplaatsing']
    y = data['Kracht']
    
    # Vind de pieken in de kracht data
    peaks, _ = find_peaks(y)
    
    # Zorg ervoor dat je de juiste indexen gebruikt voor je dataframe
    max_index = peaks[y.iloc[peaks].argmax()]
    x_max = x.iloc[max_index]
    y_max = y.iloc[max_index]
    rek_max = rek.iloc[max_index]
    
    # Interpoleer de data tot het maximum
    interpolator = interp1d(x.iloc[:max_index+1], y.iloc[:max_index+1], kind='linear')
    x_interp = np.linspace(x.iloc[0], x.iloc[max_index], 500)
    y_interp = interpolator(x_interp)
    
    # Bereken de breukenergie
    area = np.trapz(y_interp * 1000, x_interp / 1000)  # Joules (kN-mm to J)
    
    # Rek bij breuk en vormfactor
    Gc = area / (D * h)  # Gecorrigeerde breukenergie
    
    # Vormfactor berekenen
    vormfactor = (area - (0.5 * x_max * y_max)) / (0.5 * x_max * y_max)
    
    return rek_max, x_max, y_max, x_interp, y_interp, Gc, vormfactor


def define_sec_modulus(file_path, sheet, data, D, h):
    # Rek berekenen (veronderstelde formule)
    rek = 10**6 * (12 * h * 1000) / (2 * 200**2) * (data['Verplaatsing'] - data['Verplaatsing'].iloc[0])    
    
    raw_data = read_data(file_path, sheet)[1]
    process_data = data
    process_data['tijd'] = raw_data['tijd']
    process_data['rek'] = rek
    process_data['spanning'] = (1.5 * data['Kracht'] * 1000 * 200) / (h * 1000 * (D * 1000)**2)
    process_data['secantmodulus'] = process_data['spanning'] * 10**6 / process_data['rek']
    process_data = process_data.dropna()
        
    x = process_data['Verplaatsing']
    y = process_data['Kracht']
    
    # Vind pieken in kracht
    peaks, _ = find_peaks(y)
    max_index = peaks[y.iloc[peaks].argmax()]
    x_max = x.iloc[max_index]
    
    process_data['% verplaatsing'] = process_data['Verplaatsing'] / x_max
    
    idx10, idx50, idx100 = np.abs(process_data['% verplaatsing'] - 0.1).idxmin(), np.abs(process_data['% verplaatsing'] - 0.5).idxmin(), np.abs(process_data['% verplaatsing'] - 1).idxmin()
    sec_10, sec_50, sec_100 = process_data['secantmodulus'].loc[idx10], process_data['secantmodulus'].loc[idx50], process_data['secantmodulus'].loc[idx100]
    
    return sec_10, sec_50, sec_100, process_data
    

def make_plot(ax, borehole: str, originele_data, xmean, ymean, final_line, gecorrigeerde_data,
                x_max, y_max, x_interp, y_interp, G):
    ax.plot(originele_data['Verplaatsing'], originele_data['Kracht'], '.', label='Originele data', color='grey')
    ax.plot(xmean, ymean, linestyle='dashed', linewidth=1, color='grey')
    
    if final_line:
        ax.plot(final_line[0], final_line[1], linestyle='-.', color='purple', label='Lineaire fit')
    
    ax.plot(gecorrigeerde_data['Verplaatsing'], gecorrigeerde_data['Kracht'], '.', label='Gecorrigeerde data', color='orange')
    ax.plot(xmean - gecorrigeerde_data['Verplaatsing'].min(), ymean, color='black', linewidth=1)
    ax.plot(x_max, y_max, 'o', color='red', label='Max. Kracht')
    ax.fill_between(x_interp, y_interp, alpha=0.1, color='blue')
    
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 5.5)
    ax.set_xlabel('Verplaatsing [mm]')
    ax.set_ylabel('Kracht [kN]')
    ax.set_title(f'{borehole} - Gc: {G:.2f} J/m2')
    ax.legend()
    ax.grid(alpha=0.5)
    
    
def plot_graph(ax, file_path, sheet):
    originele_data, raw_data = read_data(file_path, sheet)
    D, h, strength, v = read_parameters(file_path, sheet)
    xmean = originele_data['Verplaatsing'].rolling(8).mean()
    ymean = originele_data['Kracht'].rolling(8).mean()
    max_index = ymean.idxmax()
    final_line, rc, intercept, _ = calc_linear_fit(xmean, ymean, max_index)

    gecorrigeerde_data = originele_data.copy()
    gecorrigeerde_data = correct_data(gecorrigeerde_data, rc, intercept)

    rek_max, x_max, y_max, x_interp, y_interp, Gc, vormfactor = calc_fracture_data(gecorrigeerde_data, D, h)

    make_plot(ax, sheet, originele_data, xmean, ymean, final_line, gecorrigeerde_data,
                x_max, y_max, x_interp, y_interp, Gc)
    
    
def make_table_summary_data(file_path, grafiektitel):
    sheet_lijst = []
    HR_lijst = []
    v_lijst = []
    buigtreksterkte_lijst = []
    breukenergie_lijst = []
    rek_lijst = []
    sec10_lijst = []
    sec50_lijst = []
    sec100_lijst = []
    Gc_rek_lijst = []
    Gc_rek_sig_lijst = []
    vormfactor_lijst = []
    
    f = pd.ExcelFile(file_path)
    alle_sheets = f.sheet_names
    sheetnames = alle_sheets[3:] #vanaf sheet index 3
    
    for i, sheet in enumerate(sheetnames):
        originele_data, raw_data = read_data(file_path, sheet)
        D, h, buigtreksterkte, v = read_parameters(file_path, sheet)
        xmean = originele_data['Verplaatsing'].rolling(8).mean()
        ymean = originele_data['Kracht'].rolling(8).mean()
        max_index = ymean.idxmax()
        final_line, rc, intercept, _ = calc_linear_fit(xmean, ymean, max_index)
    
        gecorrigeerde_data = originele_data.copy()
        gecorrigeerde_data = correct_data(gecorrigeerde_data, rc, intercept)
    
        rek_max, x_max, y_max, x_interp, y_interp, Gc, vormfactor = calc_fracture_data(gecorrigeerde_data, D, h)
        sec_10, sec_50, sec_100, process_data = define_sec_modulus(file_path, sheet, gecorrigeerde_data, D, h)
        
        sheet_lijst.append(sheet)
        HR_lijst.append(0)
        v_lijst.append(v)
        breukenergie_lijst.append(Gc)
        rek_lijst.append(rek_max)
        sec10_lijst.append(sec_10)
        sec50_lijst.append(sec_50)
        sec100_lijst.append(sec_100)
        vormfactor_lijst.append(vormfactor)
        buigtreksterkte_lijst.append(buigtreksterkte)
        Gc_rek_lijst.append(Gc / rek_max)
        Gc_rek_sig_lijst.append(Gc / (rek_max * buigtreksterkte))

    resultaten_df = pd.DataFrame({
    'Proefstuk': sheet_lijst,
    'HR': HR_lijst,
    'v': v_lijst,
    'sig_b': buigtreksterkte_lijst,
    'eps_b': rek_lijst,
    'Sec_10': sec10_lijst,
    'Sec_50': sec50_lijst,
    'Sec_100': sec100_lijst,    
    'G_c': breukenergie_lijst,
    'G_c_over_eps_b': Gc_rek_lijst,
    'G_c_over_eps_b_sig_b': Gc_rek_sig_lijst,
    'V_Ber': vormfactor_lijst,
    'sample_name': sheet_lijst,
    'notes': '',
    })
    tabel= resultaten_df.sort_values(by='Proefstuk',ascending=True)
    tabel.to_csv(f'_strength_summarized_data.csv', index=False)

    print(tabel)
    
    # pad = r'C:\Users\marloes.slokker\Infram BV\Infram Projecten - 23i741_KC WAB 2024 - WP 7 en 8\Uitvoering\Fase 1 - KCW (WP 7-1)\Data KCW 3PB - nagestuurde excels Kiwa KOAC\Data KCW 3PB\Fase 1'
    # tabel_naam = 'Summary_data_3PB_Delfzijl_test'
    
    # tabel.to_excel(f'{pad}\{tabel_naam}.xlsx')
    return

def make_table_raw_data(file_path):
    f = pd.ExcelFile(file_path)
    alle_sheets = f.sheet_names
    sheetnames = alle_sheets[3:]  # vanaf sheet index 3

    # Dictionary om DataFrames per sheetnaam op te slaan
    dataframes_per_sheet = {}

    for sheet in sheetnames:
        originele_data, raw_data = read_data(file_path, sheet)

        # Maak een dataframe voor deze sheet
        df = pd.DataFrame({
            'sample_name': sheet,
            't': raw_data['tijd'],
            'F': raw_data['kracht'],
            'V_org': raw_data['verplaatsing'],
            'notes': ' '
        })

        # Voeg toe aan dictionary met sheetnaam als key
        dataframes_per_sheet[sheet] = df

    # Voorbeeld: print eerste paar rijen van elke sheet
    for naam, df in dataframes_per_sheet.items():
        print(f"Proefstuk: {naam}")
        print(df.head(), "\n")

        ## save as csv
        df.to_csv(f'{naam}_strength_raw_data.csv', index=False)

    return dataframes_per_sheet

def make_table_processed_data(file_path, grafiektitel, sheet):
    f = pd.ExcelFile(file_path)
    alle_sheets = f.sheet_names
    sheetnames = alle_sheets[3:]  # vanaf sheet index 3
    
     

    # Dictionary om DataFrames per sheetnaam op te slaan
    dataframes_per_sheet = {}

    for sheet in sheetnames:
        originele_data, raw_data = read_data(file_path, sheet)
        D, h, strength, v = read_parameters(file_path, sheet)
        xmean = originele_data['Verplaatsing'].rolling(8).mean()
        ymean = originele_data['Kracht'].rolling(8).mean()
        max_index = ymean.idxmax()
        final_line, rc, intercept, _ = calc_linear_fit(xmean, ymean, max_index)

        gecorrigeerde_data = originele_data.copy()
        gecorrigeerde_data = correct_data(gecorrigeerde_data, rc, intercept)
        verplaatsing_corr = gecorrigeerde_data['Verplaatsing']
        process_data = define_sec_modulus(file_path, sheet, gecorrigeerde_data, D, h)[3]

        # Maak een dataframe voor deze sheet

        df = pd.DataFrame({
            'sample_name': sheet,
            'F': gecorrigeerde_data['Kracht'],
            # 'V_org': raw_data['verplaatsing'],
            'V_cor': verplaatsing_corr,
            'eps': process_data['rek'],
            'sig': process_data['spanning'],
            'Sec': process_data['secantmodulus'],
            'notes': '',
        })

        # Voeg toe aan dictionary met sheetnaam als key
        dataframes_per_sheet[sheet] = df

    # Voorbeeld: print eerste paar rijen van elke sheet
    for naam, df in dataframes_per_sheet.items():
        print(f"Proefstuk: {naam}")
        print(df.head(), "\n")

        df.to_csv(f'{naam}_strength_processed_data.csv', index=False)


    return dataframes_per_sheet
    


    