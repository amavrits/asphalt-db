# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 16:43:18 2025

@author: marloes.slokker
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
#from twdm import tqdm

from src.parsing.strength_parsing import read_data
from src.processing.strength_processing import plot_graph, make_table_summary_data, make_table_raw_data, make_table_processed_data


#INVOER
file_path = Path(r'c:\Users\hauth\OneDrive - Stichting Deltares\Documents\Analyse Bezwijksterkte 3pb_vak1.xlsm')
# file_path = r'C:\Users\marloes.slokker\Infram BV\Infram Projecten - 23i741_KC WAB 2024 - WP 7 en 8\Uitvoering\Fase 1 - KCW (WP 7-1)\Data KCW 3PB - nagestuurde excels Kiwa KOAC\Data KCW 3PB\Fase 1\Analyse Bezwijksterkte 3pb_1_8.xlsm'
# main_path = r'C:\Users\marloes.slokker\Infram BV\Infram Projecten - 23i740_KC WAB 2024\Uitvoering\levensduurmodel WAB'
# main_path = Path(main_path)

# project_folders = [folder for folder in main_path.iterdir()]

# file_paths = []
# for project_folder in  main_path.iterdir():
#     if project_folder.is_file():
#         continue
#     for folder in project_folder.iterdir():
#         if folder.is_file():
#             continue
#         try:
#             file_path = [f for f in folder.glob("*.xls")][0]
#             print(file_path)
#             file_paths.append(file_path)
#         except:
#             continue
    



grafiektitel = 'Vak 1 (POV Wadden)'

print(f'De resultaten van de driepuntsbuigproeven van **{grafiektitel}** zijn de volgende:')




#HOOFDSCRIPT - TABEL EN FIGUUR
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
axs = axs.flatten()

f = pd.ExcelFile(file_path)
alle_sheets = f.sheet_names
sheetnames = alle_sheets[3:]

# for i, sheet in enumerate(sheetnames):
#     plot_graph(axs[i], file_path, sheet)

make_table_raw_data(file_path, grafiektitel)    
make_table_processed_data(file_path, grafiektitel, sheetnames)    
make_table_summary_data(file_path, grafiektitel)


# plt.suptitle(grafiektitel, fontsize=15)
# plt.tight_layout(rect=[0, 0, 1, 0.98])
# plt.show()

