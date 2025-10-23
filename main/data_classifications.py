import pandas as pd

def classify_brittleness(row):
    if row['V_Ber'] < 0.10:
        brittleness = 'bros'
    elif 0.10 <= row['V_Ber'] <= 0.3:
        brittleness = 'overgang'
    else:
        brittleness = 'ductiel'
    return brittleness

def classify_material_strength(row):
    if row['sig_b'] < 3:
        strength = 'zwak'
    elif 3 <= row['sig_b'] <= 8:
        strength = 'gemiddeld'
    else:
        strength = 'sterk' 
    return strength

#construction categories before 1973, between 1973 and 2000 and after 2000
def categorize_annum(year):
    if year < 1973:
        return 'before 1973'
    elif 1973 <= year < 2000:
        return '1973-2000'
    else:
        return 'after 2000'
    

#add column of HR-categories
def categorize_hr(hr):
    if hr < 4:
        return 'HR <4%'
    elif 4 <= hr < 9:
        return 'HR 4-9%'
    else:
        return 'HR >9%'
    
#In data_per_dike_project, make a new column called heterogeneity category based on sig_b_cov
def categorize_heterogeneity(cov):
    if cov < 0.2:
        return 'Homogeen'
    elif 0.2 <= cov < 0.35:
        return 'Matig heterogeen'
    elif cov >= 0.35:
        return 'Heterogeen'