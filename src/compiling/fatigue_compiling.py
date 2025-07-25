# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
import warnings
from src.processing.fatigue_processing import make_table_raw_data
from src.processing.fatigue_processing import make_table_summary_data
from src.processing.fatigue_processing import make_table_processed_data
warnings.filterwarnings("ignore")

filename = Path(r'c:\Users\hauth\OneDrive - Stichting Deltares\projects\Asphalte Regression\DB\Vermoeiing vak 1 (1-8).xlsm')

def main():
    f = pd.ExcelFile(filename)
    sheetnames = f.sheet_names[3:]

    make_table_raw_data(filename, sheetnames)
    make_table_processed_data(filename, sheetnames)
    make_table_summary_data(filename, sheetnames)

if __name__ == "__main__":
    main()
