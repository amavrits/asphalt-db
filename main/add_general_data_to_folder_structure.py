from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
base_folder = SCRIPT_DIR.parent / "data/automated_data_new"


bernadette_file = Path(r'C:\Users\hauth\repositories\asphalte_regression\data\Database WAB - overzicht ADL28062023_BWich_selection18.8.xlsx')

bernadette_data_df = pd.read_excel(bernadette_file, sheet_name='Sheet1')




#
general_data_df = pd.read_csv(base_folder.joinpath('general_data.csv'))
master_table_df = pd.read_csv(base_folder.joinpath('master_table.csv'))


general_data_df['HR'] = general_data_df['HR'].astype(float)
general_data_df['bitumen'] = general_data_df['bitumen'].astype(float)

for idx, row in general_data_df.iterrows():
    project_id = int(row['project'].split('_')[1])
    borehole_id = int(row['borehole'][2:])

    project_data = bernadette_data_df[bernadette_data_df['Projectnummer'] == project_id]
    borehole_data = project_data[project_data['Boorkern'] == borehole_id]

    if not borehole_data.empty:
        HR = borehole_data['HR'].iloc[0]
        bitumen = borehole_data['Bitumen-gehalte NEN'].iloc[0]

        general_data_df.loc[idx, 'HR'] = HR
        general_data_df.loc[idx, 'bitumen'] = bitumen

    else:
        print("SKIPED - No data found for project_id:", project_id, "borehole_id:", borehole_id)
        general_data_df.loc[idx, 'HR'] = None
        general_data_df.loc[idx, 'bitumen'] = None




for idx, row in master_table_df.iterrows():
    project_id = int(row['project'].split('_')[1])
    borehole_id = int(row['borehole'][2:])

    project_data = bernadette_data_df[bernadette_data_df['Projectnummer'] == project_id]
    borehole_data = project_data[project_data['Boorkern'] == borehole_id]


    if not borehole_data.empty:
        construction_year = borehole_data['Aanleg-jaar'].iloc[0]
        sample_year = borehole_data['onderzoeksjaar'].iloc[0]

        master_table_df.loc[idx, 'aanlegjaar'] = construction_year
        master_table_df.loc[idx, 'onderzoeksjaar'] = sample_year

    else:
        print("SKIPED - No data found for project_id:", project_id, "borehole_id:", borehole_id)
        master_table_df.loc[idx, 'aanlegjaar'] = None
        master_table_df.loc[idx, 'onderzoeksjaar'] = None



general_data_df.to_csv(base_folder.joinpath('general_data.csv'), index=False)
master_table_df.to_csv(base_folder.joinpath('master_table.csv'), index=False)

print("Bernadette's data successfully added to the folder structure!")