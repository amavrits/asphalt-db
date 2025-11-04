import pandas as pd



def get_section_based_statistics(df_final: pd.DataFrame, dikes_and_projects: pd.DataFrame) -> pd.DataFrame:
    data_per_dike_project = pd.DataFrame.groupby(df_final, by='project_dike_id').agg({
        'v': ['mean', 'std'],
        'Sec_10': ['mean', 'std'],
        'Sec_50': ['mean', 'std'],
        'Sec_100': ['mean', 'std'], 
        'sig_b': ['mean', 'std'],
        'eps_b': ['mean', 'std'],
        'G_c': ['mean', 'std'],
        'G_c_over_eps_b': ['mean', 'std'],
        'G_c_over_eps_b_sig_b': ['mean', 'std'],
        'V_Ber': ['mean', 'std'],
        'pha_ini': ['mean', 'std'],
        'pha_50': ['mean', 'std'],
        'sig_cyc': ['mean', 'std'],
        'sig_perm': ['mean', 'std'],
        'E_ini': ['mean', 'std'],
        'E_50': ['mean', 'std'],
        'N_fat': ['mean', 'std'],
        'HR': ['mean', 'std', 'max', 'min'],
        'bitumen': ['mean', 'std'],
        'number_of_boreholes': 'count'
    })

    #flatten multiindex columns
    data_per_dike_project.columns = ['_'.join(col).strip() for col in data_per_dike_project.columns.values]

    #for each parameter make a cov column based on mean and std
    for col in data_per_dike_project.columns:
        if '_mean' in col:
            param = col.replace('_mean', '')
            data_per_dike_project[f'{param}_cov'] = data_per_dike_project[f'{param}_std'] / data_per_dike_project[col]

    #take unique for project_dike_id in dikes_and_projects
    unique_dike_projects = dikes_and_projects.drop_duplicates(subset=['project_dike_id'])

    data_per_dike_project = pd.merge(unique_dike_projects, data_per_dike_project, left_on='project_dike_id', right_index=True, how='right')
    
    data_per_dike_project.reset_index(drop=True, inplace=True)

    return data_per_dike_project
