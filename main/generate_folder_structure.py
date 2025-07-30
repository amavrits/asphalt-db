import re
import time
from collections import defaultdict

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import shutil

from src.parsing.fatigue_parsing import read_raw_fatigue, read_processed_fatigue, read_summary_fatigue
from src.parsing.strength_parsing import read_data, read_parameters
from src.processing.strength_processing import make_table_raw_data, calc_linear_fit, correct_data, define_sec_modulus, \
    calc_fracture_data


def fill_master_table_data(project: int, vak_name: str, borehole_id_list: list[int], master_table_data: list) -> list:
    for borehole_id in borehole_id_list:
        master_table_data.append(
            [f"P_{project}", f"BH{borehole_id}", f"{vak_name}"]
        )
    return master_table_data


def fill_general_table_data(project_id: int, borehole_id, general_table_data: list):
    general_table_data.append([f"P_{project_id}", f"BH{borehole_id}", f"{borehole_id}B", 0])
    general_table_data.append([f"P_{project_id}", f"BH{borehole_id}", f"{borehole_id}V", 0])
    return general_table_data


def fill_project_data_csv(base_folder: Path, project_names):
    n_projects = len(project_names)
    project_data = {
        "project_name": project_names,
        "project_code": ["BBBBB"] * n_projects,
        "date": [str(datetime.utcnow())] * n_projects,
        "notes": ["AAAA"] * n_projects
    }
    df_projects = pd.DataFrame(data=project_data)
    df_projects.to_csv(base_folder.joinpath("project_table.csv"), index=False)


def fill_dike_data_table_df(base_folder: Path, vak_dict: dict):
    dike_names = list(vak_dict.keys())
    n_dikes = len(dike_names)
    dike_data = {
        "dike_name": dike_names,
        "waterboard": ["HHNK"] * n_dikes,
        "notes": [""] * n_dikes,
    }
    df_dikes = pd.DataFrame(data=dike_data)
    df_dikes.to_csv(base_folder.joinpath("dike_table.csv"), index=False)


def fill_borehole_data_csv(borehole_path: Path, borehole_name: str):
    borehole_data = {
        "borehole_name": f"{borehole_name}",
        "collection_date": str(datetime.utcnow()),
        "notes": ["AAAA"],
        "X_coord": 0,
        "Y_coord": 0,
    }
    with open(borehole_path.joinpath("borehole_data.json"), "w") as f:
        json.dump(borehole_data, f, indent=4)


def fill_sample_data_csv(borehole_path: Path, borehole_id, strength_file: Path):
    D, h, strength, v = read_parameters(strength_file, f"{borehole_id}B")

    sample_data = {f"{borehole_id}B": {
        "depth": 0,
        "thickness": D,
        "height": h,
        "strength": strength,
        "v": v,
        "notes": [
            "DDDDDD"
        ]
    },
        f"{borehole_id}V": {
            "depth": 0,
            "thickness": 0.0,
            "height": 0.0,
            "fatigue": 0.0,
            "v": 0.0,
            "notes": [
                "DDDDDD"
            ]
        }

    }

    with open(borehole_path / "sample_data.json", "w") as f:
        json.dump(sample_data, f, indent=4)


def fill_strength_data_csv(borehole_path: Path, sample_name: str, file_path: Path):
    """
    Fill the all three strength csv: Raw, processed and summarized data.
    :param borehole_path: Path to the borehole folder where the data will be saved
    :param sample_name:
    :param file_path: Path to the excel file containing the data
    :return:
    """
    test_path = borehole_path / f"strength"
    test_path.mkdir(exist_ok=True, parents=True)

    # 1. Get raw data csv
    originele_data, raw_data = read_data(file_path, sample_name)

    # Maak een dataframe voor deze sheet
    df_raw = pd.DataFrame({
        'sample_name': sample_name,
        't': raw_data['tijd'],
        'F': raw_data['kracht'],
        'V_org': raw_data['verplaatsing'],
        'notes': ' '
    })
    df_raw.to_csv(test_path / f"raw_data.csv", index=False)

    # 2. Get processed data csv
    D, h, strength, v = read_parameters(file_path, sample_name)
    xmean = originele_data['Verplaatsing'].rolling(8).mean()
    ymean = originele_data['Kracht'].rolling(8).mean()
    max_index = ymean.idxmax()
    final_line, rc, intercept, _ = calc_linear_fit(xmean, ymean, max_index)

    gecorrigeerde_data = originele_data.copy()
    gecorrigeerde_data = correct_data(gecorrigeerde_data, rc, intercept)
    verplaatsing_corr = gecorrigeerde_data['Verplaatsing']
    process_data = define_sec_modulus(file_path, sample_name, gecorrigeerde_data, D, h)[3]

    df_processed = pd.DataFrame({
        'sample_name': sample_name,
        'F': gecorrigeerde_data['Kracht'],
        # 'V_org': raw_data['verplaatsing'],
        'V_cor': verplaatsing_corr,
        'eps': process_data['rek'],
        'sig': process_data['spanning'],
        'Sec': process_data['secantmodulus'],
        'notes': '',
    })
    df_processed.to_csv(test_path / f"processed_data.csv", index=False)

    # 3. Get summarized data csv
    rek_max, x_max, y_max, x_interp, y_interp, Gc, vormfactor = calc_fracture_data(gecorrigeerde_data, D, h)
    sec_10, sec_50, sec_100, process_data = define_sec_modulus(file_path, sample_name, gecorrigeerde_data, D, h)

    df_summarized = pd.DataFrame({
        'sample_name': sample_name,
        'HR': 0,  # TODO ??

        'v': v,
        'sig_b': strength,  # TODO: find a better name
        'eps_b': rek_max,
        'Sec_10': sec_10,
        'Sec_50': sec_50,
        'Sec_100': sec_100,
        'G_c': Gc,
        'G_c_over_eps_b': Gc / rek_max,
        'G_c_over_eps_b_sig_b': Gc / (rek_max * strength),
        'V_Ber': vormfactor,
        'notes': '',
    }, index=[0])
    tabel = df_summarized.sort_values(by='sample_name', ascending=True)
    tabel.to_csv(test_path / f"summarized_data.csv", index=False)


def fill_fatigue_data_csv(borehole_path, sample_name: str, file_path: Path):
    """
    Fill the fatigue data csv files: raw_data.csv, processed_data.csv and summarized_data.csv.
    :param borehole_path: path to the borehole folder where the data will be saved
    :param sample_name:
    :param file_path: Path to the excel file containing the data
    :return:
    """
    test_path = borehole_path / f"fatigue"
    test_path.mkdir(exist_ok=True, parents=True)

    # 1. Get raw data csv
    raw_data = read_raw_fatigue(file_path, sample_name)

    df_raw = pd.DataFrame({
        'sample_name': sample_name,
        'N': 0,  # What is N ???
        'MaximumStroke': raw_data['MaximumStroke'],
        'MinimumStroke': raw_data['MinimumStroke'],
        'PeakToPeakStroke': raw_data['PeakToPeakStroke'],
        'MaximumLoad': raw_data['MaximumLoad'],
        'PeakToPeakLoad': raw_data['PeakToPeakLoad'],
        'InPhaseModulus': raw_data['InPhaseModulus'],
        'OutPhaseModulus': raw_data['OutPhaseModulus'],
        'notes': ''
    })
    df_raw.to_csv(test_path / f"raw_data.csv", index=False)

    # 2. Get processed data csv
    processed_data = read_processed_fatigue(file_path, sample_name)

    df_processed = pd.DataFrame({
        'sample_name': sample_name,
        'N': processed_data['N'],
        'eps_cycl': processed_data['eps_cycl'],
        'eps_perm': processed_data['eps_perm'],
        'sig_cyc': processed_data['sig_cyc'],
        'sig_perm': processed_data['sig_perm'],
        'E_dyn': processed_data['E_dyn'],
        'pha': processed_data['pha'],
        'notes': '',
    })

    df_processed = df_processed.dropna(subset=['eps_cycl'])  # Cut df where eps_cycl is NaN
    df_processed.to_csv(test_path / f"processed_data.csv", index=False)

    # 3. Get summarized data csv
    pha_ini, pha_50, sig_cyc, sig_perm, E_ini, E_50, N_fat = read_summary_fatigue(file_path, sample_name)

    df_summarized = pd.DataFrame({
        'sample_name': sample_name,
        'pha_ini': pha_ini,
        'pha_50': pha_50,
        'sig_cyc': sig_cyc,
        'sig_perm': sig_perm,
        'E_ini': E_ini,
        'E_50': E_50,
        'N_fat': N_fat
    }, index=[0])

    df_summarized = df_summarized.sort_values(by='sample_name', ascending=True)
    df_summarized.to_csv(test_path / f"summarized_data.csv", index=False)


def fill_stiffness_data_csv(borehole_path):
    test_path = borehole_path / f"stiffness"
    test_path.mkdir(exist_ok=True, parents=True)

# def add_test_data_json():
#     test_data = {
#         "str_appratus": "A",
#         "ftg_appratus": "B",
#         "stiff_appratus": "C",
#         "notes": ["DDDDDD"],
#     }
#     # TODO
#     with open(test_path / "test_data.json", "w") as f:
#         json.dump(test_data, f, indent=4)


if __name__ == "__main__":
    tic = time.time()
    n_projects = 1

    SCRIPT_DIR = Path(__file__).parent
    base_folder = SCRIPT_DIR.parent / "data/automated_data"
    input_files_folder = Path(r'c:\Users\hauth\OneDrive - Stichting Deltares\projects\Asphalte Regression\DB\data1') # make the path a env variable
    if base_folder.is_dir():
        shutil.rmtree(base_folder)
    base_folder.mkdir(exist_ok=True, parents=True)

    fill_project_data_csv(base_folder, [f"P_{i}" for i in range(1, n_projects + 1)])
    for project in range(1, n_projects + 1):

        # Grouping by vak
        vak_dict = {}
        for file in input_files_folder.iterdir():
            filename = file.name
            vak = filename.split('_')[0]
            if vak not in vak_dict:
                vak_dict[vak] = {}
            if "Analyse Bezwijksterkte" in filename:
                vak_dict[vak]["strength"] = file
            elif "Vermoeiing" in filename:
                vak_dict[vak]["fatigue"] = file
            elif 'master' in filename:
                continue

        # Loop over all the dike
        master_table_data = []
        general_table_data = []

        fill_dike_data_table_df(base_folder, vak_dict)

        for vak_name, vak_files in vak_dict.items():

            strength_file = vak_files.get("strength")
            fatigue_file = vak_files.get("fatigue")

            sample_name_strength = pd.ExcelFile(strength_file).sheet_names[3:]
            sample_name_fatigue = pd.ExcelFile(fatigue_file).sheet_names[3:]

            # Validation of sample names for the current vak
            # if len(sample_name_strength) != len(sample_name_fatigue):
            #     raise ValueError("Number of samples in strength and fatigue files do not match.")

            # nums_b = {re.match(r'(\d+)B$', item).group(1) for item in sample_name_strength if re.match(r'\d+B$', item)}
            # nums_v = {re.match(r'(\d+)V$', item).group(1) for item in sample_name_fatigue if re.match(r'\d+V$', item)}

            # if len(nums_b) != len(nums_v):
            #     raise ValueError("Borehole ids are not matching")
            # n_bhs = len(sample_name_strength)
            strength_sample_ids = {re.match(r'(\d+)', item).group(1) for item in sample_name_strength}
            fatigue_sample_ids = {re.match(r'(\d+)', item).group(1) for item in sample_name_fatigue}
            borehole_ids = list(set(list(strength_sample_ids) + list(fatigue_sample_ids)))


            fill_master_table_data(project, vak_name, list(borehole_ids), master_table_data)
            #There can be one borehole without strength because the test was bad or something.



            for borehole_id in borehole_ids:
                borehole_name = f"BH{borehole_id}"

                borehole_path = base_folder.joinpath(f"P_{project}", borehole_name)
                borehole_path.mkdir(exist_ok=True, parents=True)

                fill_borehole_data_csv(borehole_path, borehole_name)
                fill_sample_data_csv(borehole_path, borehole_id, strength_file)
                fill_general_table_data(project, borehole_id, general_table_data)

                fill_strength_data_csv(borehole_path, f"{borehole_id}B", strength_file)
                fill_fatigue_data_csv(borehole_path, f"{borehole_id}V", fatigue_file)
                fill_stiffness_data_csv(borehole_path)

        master_table_df = pd.DataFrame(master_table_data, columns=["project", "borehole", "dike"])
        master_table_df.to_csv(base_folder.joinpath("master_table.csv"), index=False)

        general_data_df = pd.DataFrame(general_table_data, columns=["project", "borehole", "sample", "e"])
        general_data_df = general_data_df.drop_duplicates(subset=["project", "borehole", "sample"])

        general_data_df.to_csv(base_folder.joinpath("general_data.csv"), index=False)

    toc = time.time()
    print(f"Time taken: {toc - tic:.2f} seconds")
