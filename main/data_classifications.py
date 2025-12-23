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
        return 'voor 1973'
    elif 1973 <= year < 2000:
        return '1973-2000'
    else:
        return 'na 2000'
    

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
    

def classifications_wrapper(df_final: pd.DataFrame, data_per_dike_project: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df_final['HR_category'] = df_final['HR'].apply(categorize_hr)    
    df_final['brittleness'] = df_final.apply(classify_brittleness, axis=1)
    df_final['material_class'] = df_final.apply(classify_material_strength, axis=1)
    df_final['construction_year_category'] = df_final.construction_year.apply(categorize_annum)

    data_per_dike_project['heterogeneity_category'] = data_per_dike_project['sig_b_cov'].apply(categorize_heterogeneity)
    data_per_dike_project['construction_year_category'] = data_per_dike_project.construction_year.apply(categorize_annum)
    
    #heterogeniteitscategorie: maak een dict van project_dike_id en de corresponderende heterogeniteitscategorie o.b.v. data_per_dike_project
    heterogeneity_dict = data_per_dike_project.set_index('project_dike_id')['heterogeneity_category'].to_dict()
    heterogeneity_dict
    #toevoegen aan df_final
    
    df_final['heterogeneity_category'] = df_final['project_dike_id'].map(heterogeneity_dict)
    return df_final, data_per_dike_project