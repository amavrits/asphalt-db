# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#from twdm import tqdm
from sqlalchemy import create_engine
from sqlalchemy.types import Float, String, Integer

from src.parsing.fatigue_parsing import read_raw_fatigue
from src.parsing.fatigue_parsing import read_summary_fatigue
from src.processing.fatigue_processing import make_table_raw_data
from src.processing.fatigue_processing import make_table_summary

filename = r'C:\Users\inge.brijker\Infram BV\Infram Projecten - 23i740_KC WAB 2024\Uitvoering\levensduurmodel WAB\1600982\Vermoei\Analyse Vermoeiing 3pb.xlsm'
# filename = 'Vermoeiing vak 1 (1-8) Versie2.xlsm'
f = pd.ExcelFile(filename)

alle_sheets = f.sheet_names
sheetnames = alle_sheets[3:]


    
   
# TODO 
# function die berekende data roept

for i, sheet in enumerate(sheetnames):
    read_raw_fatigue(filename, sheet)
    read_summary_fatigue(filename, sheet)
    
read_summary_fatigue(filename, sheet)

ruwe_data = make_table_raw_data(filename, sheetnames)
print(f"Ruwe data uit sheet {sheetnames}:\n", ruwe_data.head())

summary = read_summary_fatigue(filename, sheetnames)
print(f"Samenvatting uit sheet {sheetnames}:\n", summary)



if __name__ == "__main__":
    pass

