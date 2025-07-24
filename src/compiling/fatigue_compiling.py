# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#from twdm import tqdm
from sqlalchemy import create_engine
from sqlalchemy.types import Float, String, Integer

from src.parsing.fatigue_parsing import read_raw_fatigue
from src.parsing.fatigue_parsing import read_processed_fatigue
from src.parsing.fatigue_parsing import read_summary_fatigue
from src.processing.fatigue_processing import make_table_raw_data
from src.processing.fatigue_processing import make_table_summary_data
from src.processing.fatigue_processing import make_table_processed_data

filename = r'C:\Users\inge.brijker\Infram BV\Infram Projecten - 23i740_KC WAB 2024\Uitvoering\levensduurmodel WAB\1600982\Vermoei\Analyse Vermoeiing 3pb.xlsm'

def main():
    f = pd.ExcelFile(filename)
    sheetnames = f.sheet_names[3:]

    # Roep functies aan voor alle sheets in één keer
    make_table_raw_data(filename, sheetnames)
    make_table_processed_data(filename, sheetnames)
    make_table_summary_data(filename, sheetnames)

if __name__ == "__main__":
    main()
