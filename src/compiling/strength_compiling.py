# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 16:43:18 2025

@author: marloes.slokker
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#from twdm import tqdm

#from src.parsing.strength_parsing import read_data
from src.processing.strength_processing import plot_graph, make_table




#INVOER
file_path = r'C:\Users\marloes.slokker\Infram BV\Infram Projecten - 23i741_KC WAB 2024 - WP 7 en 8\Uitvoering\Fase 1 - KCW (WP 7-1)\Data KCW 3PB - nagestuurde excels Kiwa KOAC\Data KCW 3PB\Fase 1\Analyse Bezwijksterkte 3pb_1_8.xlsm'
grafiektitel = 'Delfzijl 1-8'

print(f'De resultaten van de driepuntsbuigproeven van **{grafiektitel}** zijn de volgende:')




#HOOFDSCRIPT - TABEL EN FIGUUR
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
axs = axs.flatten()

f = pd.ExcelFile(file_path)
alle_sheets = f.sheet_names
sheetnames = alle_sheets[3:]

for i, sheet in enumerate(sheetnames):
    plot_graph(axs[i], file_path, sheet)
    
make_table(file_path, grafiektitel)

plt.suptitle(grafiektitel, fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
    
    