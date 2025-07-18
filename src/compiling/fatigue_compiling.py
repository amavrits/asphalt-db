# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#from twdm import tqdm
from sqlalchemy import create_engine
from sqlalchemy.types import Float, String, Integer

from src.parsing.fatigue_parsing import read_fatigue 
from src.processing.fatigue_processing import calc_permanent_strain

# TODO
db_url = "postgresql://username:password@localhost:5432/your_database"


filename = 'Vermoeiing vak 1 (1-8) Versie2.xlsm'   # Voer hier het pad naar je Excelbestand in
f = pd.ExcelFile(filename)
alle_sheets = f.sheet_names
sheetname = alle_sheets[3:] #vanaf sheet index 4

def data_toevoegen_samenvatting(filename, sheet):
    sheet_lijst = []
    fasehoek_lijst = []
    permanente_rek_lijst = []
    Spanning_C_lijst = []
    Spanning_P_lijst = []
    Stijfheid_lijst = []
    Nfat_lijst = []

#dit werkt nog niet omdat aflezing nu apart staat binnen parsing
    for sheet in sheetname:
        fasehoek = read_fatigue(filename, sheet)[0]
        permanente_rek = calc_permanent_strain(filename, sheet)
        Spanning_C = read_fatigue(filename, sheet)[1]
        Spanning_P = read_fatigue(filename, sheet)[2]
        Stijfheid = read_fatigue(filename, sheet)[3]
        Nfat = read_fatigue(filename, sheet)[4]

        sheet_lijst.append(sheet)
        fasehoek_lijst.append(fasehoek)
        permanente_rek_lijst.append(permanente_rek)
        Spanning_C_lijst.append(Spanning_C)
        Spanning_P_lijst.append(Spanning_P)
        Stijfheid_lijst.append(Stijfheid)
        Nfat_lijst.append(Nfat)
        
    resultaten_df = pd.DataFrame({
        'Sheetnaam': sheet_lijst,
        'Fasehoek [°]': fasehoek_lijst,
        'Permanente Rek [µm/m]': permanente_rek_lijst,
        'Cyclische Spanning [MPa]': Spanning_C_lijst,
        'Permanente Spanning [MPa]': Spanning_P_lijst,
        'Stijfheid [MPa]': Stijfheid_lijst,
        'N_fat [-]': Nfat_lijst
    })
    tabel_samenvatting_data = resultaten_df.sort_values(by='Sheetnaam', ascending=True)
    print (tabel_samenvatting_data)
    

data_toevoegen_samenvatting (filename, sheetname)