# -*- coding: utf-8 -*-

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from src.processing.stiffness_processing import make_table_raw_data
from src.processing.stiffness_processing import make_table_summary_data

filename = r'C:\Users\inge.brijker\Infram BV\Infram Projecten - 23i740_KC WAB 2024\Uitvoering\levensduurmodel WAB\1900384\Edyn\vak 1 Stijfheid (1-8).xlsm'

def main():
    f = pd.ExcelFile(filename)
    sheetnames = f.sheet_names[3:]

    make_table_raw_data(filename, sheetnames)
    make_table_summary_data(filename, sheetnames)

if __name__ == "__main__":
    main()

