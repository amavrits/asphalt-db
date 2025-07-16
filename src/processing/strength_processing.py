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
from src.parsing.strength_parsing import read_data


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
    

def make_plot(ax, sheet, originele_data, xmean, ymean, final_line, gecorrigeerde_data,
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
    ax.set_title(f'{sheet} - Gc: {G:.2f} J/m2')
    ax.legend()
    ax.grid(alpha=0.5)
    
    
def analyse_sheet(ax, file_path, sheet):
    originele_data, D, h, strength = read_data(file_path, sheet)
    xmean = originele_data['Verplaatsing'].rolling(8).mean()
    ymean = originele_data['Kracht'].rolling(8).mean()
    max_index = ymean.idxmax()
    final_line, rc, intercept, _ = calc_linear_fit(xmean, ymean, max_index)

    gecorrigeerde_data = originele_data.copy()
    gecorrigeerde_data = correct_data(gecorrigeerde_data, rc, intercept)

    rek_max, x_max, y_max, x_interp, y_interp, Gc, vormfactor = calc_fracture_data(gecorrigeerde_data, D, h)

    make_plot(ax, sheet, originele_data, xmean, ymean, final_line, gecorrigeerde_data,
                x_max, y_max, x_interp, y_interp, Gc)
    
    
def make_table(file_path, grafiektitel):
    sheet_lijst = []
    buigtreksterkte_lijst = []
    breukenergie_lijst = []
    rek_lijst = []
    vormfactor_lijst = []
    
    f = pd.ExcelFile(file_path, engine="xlrd")
    alle_sheets = f.sheet_names
    sheetnames = alle_sheets[3:] #vanaf sheet index 3
    
    for i, sheet in enumerate(sheetnames):
        originele_data, D, h, buigtreksterkte = read_data(file_path, sheet)
        xmean = originele_data['Verplaatsing'].rolling(8).mean()
        ymean = originele_data['Kracht'].rolling(8).mean()
        max_index = ymean.idxmax()
        final_line, rc, intercept, _ = calc_linear_fit(xmean, ymean, max_index)
    
        gecorrigeerde_data = originele_data.copy()
        gecorrigeerde_data = correct_data(gecorrigeerde_data, rc, intercept)
    
        rek_max, x_max, y_max, x_interp, y_interp, Gc, vormfactor = calc_fracture_data(gecorrigeerde_data, D, h)
        
        sheet_lijst.append(sheet)
        breukenergie_lijst.append(Gc)
        rek_lijst.append(rek_max)
        vormfactor_lijst.append(vormfactor)
        buigtreksterkte_lijst.append(buigtreksterkte)


    resultaten_df = pd.DataFrame({
    'Sheetnaam': sheet_lijst,
    'Buigtreksterkte [MPa]': buigtreksterkte_lijst,
    'Breukenergie [J/m2]': breukenergie_lijst,
    'Rek Bij Breuk [µm/m]': rek_lijst,
    'Vormfactor [-]': vormfactor_lijst})
    tabel= resultaten_df.sort_values(by='Sheetnaam',ascending=True)
    print(tabel)  
    #tabel.to_excel('Resultaten 3PB 25_32.xlsx')
    return

    