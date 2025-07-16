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

from src.parsing.strength_parsing import read_data
from src.processing.strength_processing import correct_data, calc_fracture_data, calc_linear_fit, make_plot, analyse_sheet, make_table

import os
#INVOER
file_path = r'C:\Users\marloes.slokker\Infram BV\Infram Projecten - 23i741_KC WAB 2024 - WP 7 en 8\Uitvoering\Fase 1 - KCW (WP 7-1)\Data KCW 3PB - nagestuurde excels Kiwa KOAC\Data KCW 3PB\Fase 1\Analyse Bezwijksterkte 3pb_1_8.xlsm'
grafiektitel = 'Vak 7'
print(os.path.exists(path=file_path))
print(f'De resultaten van de driepuntsbuigproeven van **{grafiektitel}** zijn de volgende:')

fig, axs = plt.subplots(2, 4, figsize=(20, 10))
axs = axs.flatten()

f = pd.ExcelFile(file_path)
alle_sheets = f.sheet_names
sheetnames = alle_sheets[3:]

for i, sheet in enumerate(sheetnames):
    analyse_sheet(axs[i], file_path, sheet)
    
make_table(file_path, grafiektitel)

plt.suptitle(grafiektitel, fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
    
    