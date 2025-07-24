# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 13:55:17 2025

@author: marloes.slokker
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.parsing.parsing_tables import make_headtable
from src.processing.processing_tables import make_table2

# File path
pad = r'C:\Users\marloes.slokker\Infram BV\Infram Projecten - 23i740_KC WAB 2024\Uitvoering\levensduurmodel WAB'
bestandsnaam = 'Database Asfalt Excel.xlsx'

file_path = f'{pad}\{bestandsnaam}'

# Projectnummers
projectnummers = ['048304', '0601831', '0700462/0702493', '0801782', '0802158', '0803318', '0900262', '0901480', '0901602', '0901858', '0902601', 
                  '092633', '1000038', '1000377', '1103367', '1202159', '1203379', '1300348', '1400863', '1600982', '1604257', '1604441', '1700160',
                  '1702837', '1702899', '1900384', '1900559', '1901142', '1903808', '1903877', '2000204', '2001233', '2001977', '2003106', '2004437',
                  '2100120', '2100513', '2200207', '2202064', '2202263', '2300963', '2301408']

# Dijktraject
dijktraject = 'n.t.b.'

make_headtable(file_path, projectnummers, dijktraject)
make_table2(file_path, projectnummers)