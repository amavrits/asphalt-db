import re
import time
from collections import defaultdict

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import shutil

from main.generate_template_master_table import input_files_folder
from src.parsing.fatigue_parsing import read_raw_fatigue, read_processed_fatigue, read_summary_fatigue
from src.parsing.stiffness_parsing import read_raw_stiffness
from src.parsing.strength_parsing import read_data, read_parameters
from src.processing.strength_processing import make_table_raw_data, calc_linear_fit, correct_data, define_sec_modulus, \
    calc_fracture_data


def fill_master_table_data(project: int, vak_name: str, borehole_id_list: list[int], master_table_data: list) -> list:
    """
    The columns of the master table are: project, borehole, dijk, aanlegjaar, onderzoekjaar
    :param project:
    :param vak_name:
    :param borehole_id_list:
    :param master_table_data:
    :return:
    """

    # fillers for aanlegjaar and onderzoekjaar, these will be filled later with the data from Bernadette
    for borehole_id in borehole_id_list:
        master_table_data.append(
            [f"P_{project}", f"BH{borehole_id}", f"{vak_name}", 1900, 1900]
        )
    return master_table_data


def fill_general_table_data(project_id: int, borehole_id, sample_name_strength: str, sample_name_fatigue: str,
                            sample_name_stiffness: str,
                            general_table_data: list):
    """
    Columns of the general table are: project, borehole, HR, bitumen
    :param project_id:
    :param borehole_id:
    :param sample_name_strength:
    :param sample_name_fatigue:
    :param sample_name_stiffness:
    :param general_table_data:
    :return:
    """

    # Fillers for HR and bitumen, these will be filled later with the data from Bernadette
    general_table_data.append([f"P_{project_id}", f"BH{borehole_id}", sample_name_strength, 0, 0])
    general_table_data.append([f"P_{project_id}", f"BH{borehole_id}", sample_name_fatigue, 0, 0])
    general_table_data.append([f"P_{project_id}", f"BH{borehole_id}", sample_name_stiffness, 0, 0])
    return general_table_data


def fill_project_data_csv(base_folder: Path, project_names: list[int]):
    project_data = []
    for project_id in project_names:
        project_folder = base_folder.joinpath(f"P_{project_id}")
        project_folder.mkdir(exist_ok=True, parents=True)
        project_data.append({
            "project_name": f"P_{project_id}",
            "project_code": f"{project_id}",
            "date": str(datetime.utcnow()),  # TODO: remove.
            "notes": "AAAA"
        })

    df_projects = pd.DataFrame(data=project_data)
    df_projects.to_csv(base_folder.joinpath("project_table.csv"), index=False)


def fill_dike_data_table_df(vak_dict: dict, dike_data):
    dike_names = list(vak_dict.keys())
    for dike_name in dike_names:
        dike_data.append({
            "dike_name": dike_name,
            "waterboard": "HHNK",
            "notes": "",
        })


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


def fill_sample_data_csv(borehole_path: Path, sample_name_strength: str, sample_name_fatigue: str,
                         sample_name_stiffness: str, strength_file: Path):
    D, h, strength, v = read_parameters(strength_file, sample_name_strength)

    sample_data = {sample_name_strength: {
        "depth": 0,
        "thickness": D,
        "height": h,
        "strength": strength,
        "v": v,
        "notes": [
            "DDDDDD"
        ]
    },
        sample_name_fatigue: {
            "depth": 0,
            "notes": [
                "DDDDDD"
            ]
        },
        sample_name_stiffness: {
            "depth": 0,
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


def fill_stiffness_data_csv(borehole_path, sample_name: str, stiffness_file: Path):
    """
    Fill the stiffness data csv files: raw_data.csv and summarized_data.csv.
    :param borehole_path:
    :param sample_name:
    :param stiffness_file:
    :return:
    """
    test_path = borehole_path / f"stiffness"
    test_path.mkdir(exist_ok=True, parents=True)

    raw_data, temp = read_raw_stiffness(stiffness_file, sample_name)

    df_raw = pd.DataFrame({
        'sample_name': sample_name,
        'f': raw_data['f'],
        'eps': raw_data['eps'],
        'E_dyn': raw_data['E_dyn'],
        'pha': raw_data['pha'],
        'notes': '',
    })

    df_raw.to_csv(test_path / f"raw_data.csv", index=False)

    # Filter rijen waar f == 10
    f10_data = raw_data[raw_data['f'] == 10]

    if not f10_data.empty:
        if len(f10_data) == 1:
            E_dyn_value = f10_data['E_dyn'].iloc[0]
        else:
            E_dyn_value = f10_data['E_dyn'].mean()
    else:
        E_dyn_value = np.nan

    df_summarized = pd.DataFrame({
        'sample_name': sample_name,
        'E_dyn_summary (f=10)': E_dyn_value,
        'Temp (°C)': temp
    }, index=[0])

    df_summarized = df_summarized.sort_values(by='sample_name', ascending=True)
    df_summarized.to_csv(test_path / f"summarized_data.csv", index=False)


def get_sample_names_from_sheet(file: Path) -> list[str]:
    """
    Extract sample names from the Excel file, excluding specific sheets: Invoer, Resultaten, Grafieken, ORG,...
    Unfortunately, it is not reliable to simply read the tabs and exclude only the first 3. Also not reliable to suppose
    a sample starts with a figure or a letter.
    :param file:
    :return:
    """
    samples = pd.ExcelFile(file).sheet_names[
              3:]  # hopefully the first 3 sheets are not samples and should be discarded?

    # remove string like ORG, Invoer, Resultaten, Grafieken, .....
    samples = [str(sample) for sample in samples if not re.match(
        r'^(ORG|Invoer|Resultaten|Grafieken|Org|Vermoeiingslijn|Layout voor rapportage|Resultaat|Blad1)$', sample)]
    return samples


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

    SCRIPT_DIR = Path(__file__).parent
    base_folder = SCRIPT_DIR.parent / "data/automated_data_new"
    input_files_folder = Path(
        r'c:\Users\hauth\OneDrive - Stichting Deltares\projects\Asphalte Regression\DB\data3')  # make the path a env variable
    if base_folder.is_dir():
        shutil.rmtree(base_folder)
    base_folder.mkdir(exist_ok=True, parents=True)

    master_table = pd.read_excel(input_files_folder.joinpath("master_table.xlsx"))
    projects_ids = master_table['project'].dropna().unique().astype(int).tolist()

    fill_project_data_csv(base_folder, projects_ids)

    project_dict = {}
    # Loop over all the dijk
    master_table_data = []
    general_table_data = []
    dike_table_data = []
    for project_id in projects_ids:
        project_dict[f"P_{project_id}"] = {}

        # Group all the files by vak
        vak_dict_mapping = {}
        project_master_table = master_table[master_table['project'] == project_id]

        for _, row in project_master_table.iterrows():
            filename = row['filename']
            vak = row['dijk']
            if vak not in vak_dict_mapping:
                vak_dict_mapping[vak] = {}
            if "bezwijksterkte" in filename.lower():
                vak_dict_mapping[vak]["strength"] = filename
            elif "vermoeiing" in filename.lower():
                vak_dict_mapping[vak]["fatigue"] = filename
            elif "stijfheid" in filename.lower():
                vak_dict_mapping[vak]["stiffness"] = filename
            elif 'master' in filename:
                continue

        fill_dike_data_table_df(vak_dict_mapping, dike_table_data)

        for vak_name, vak_files in vak_dict_mapping.items():

            strength_file = input_files_folder.joinpath(vak_files.get("strength"))
            fatigue_file = input_files_folder.joinpath(vak_files.get("fatigue"))
            stiffness_file = input_files_folder.joinpath(vak_files.get("stiffness"))

            sample_name_strength = get_sample_names_from_sheet(strength_file)
            sample_name_fatigue = get_sample_names_from_sheet(fatigue_file)
            sample_name_stiffness = get_sample_names_from_sheet(stiffness_file)


            def make_mapping(strength_list, fatigue_list, stiffness_list):
                mapping = defaultdict(lambda: {"strength": [], "fatigue": [], "stiffness": []})

                # Add strength samples
                for s in strength_list:
                    m = re.search(r'\d+', s)
                    if m:  # only process if a number exists
                        num = int(m.group(0))
                        mapping[num]["strength"].append(s)

                # Add fatigue samples
                for f in fatigue_list:
                    m = re.search(r'\d+', f)
                    if m:  # only process if a number exists
                        num = int(m.group(0))
                        mapping[num]["fatigue"].append(f)

                # add stiffness samples
                for s in stiffness_list:
                    m = re.search(r'\d+', s)
                    if m:
                        num = int(m.group(0))
                        mapping[num]["stiffness"].append(s)

                return dict(mapping)


            sample_names_mapping_dict = make_mapping(sample_name_strength, sample_name_fatigue, sample_name_stiffness)

            borehole_ids = list(sample_names_mapping_dict.keys())
            borehole_ids.sort()

            fill_master_table_data(project_id, vak_name, borehole_ids, master_table_data)
            # There can be one borehole without strength because the test was bad or something.

            for borehole_id in borehole_ids:
                borehole_name = f"BH{borehole_id}"
                strength_sample_name = sample_names_mapping_dict[borehole_id]["strength"][0]
                fatigue_sample_name = sample_names_mapping_dict[borehole_id]["fatigue"][0]
                stiffness_sample_name = sample_names_mapping_dict[borehole_id]["stiffness"][0]

                if len(sample_names_mapping_dict[borehole_id]["fatigue"]) > 1:
                    raise Exception(
                        f"Sample {sample_names_mapping_dict[borehole_id]['fatigue']} has multiple fatigue samples (project {project_id}). Please modify the file so that there is only one sample per borehole.")

                borehole_path = base_folder.joinpath(f"P_{project_id}", borehole_name)
                borehole_path.mkdir(exist_ok=True, parents=True)

                fill_borehole_data_csv(borehole_path, borehole_name)
                fill_sample_data_csv(borehole_path, strength_sample_name, fatigue_sample_name, stiffness_sample_name,
                                     strength_file)
                fill_general_table_data(project_id, borehole_id, strength_sample_name, fatigue_sample_name,
                                        stiffness_sample_name, general_table_data)

                # TODO: what if multiple sheet for one borehole id??
                fill_strength_data_csv(borehole_path, strength_sample_name, strength_file)
                fill_fatigue_data_csv(borehole_path, fatigue_sample_name, fatigue_file)
                fill_stiffness_data_csv(borehole_path, stiffness_sample_name, stiffness_file)

    master_table_df = pd.DataFrame(master_table_data,
                                   columns=["project", "borehole", "dijk", "aanlegjaar", "onderzoeksjaar"])

    general_data_df = pd.DataFrame(general_table_data, columns=["project", "borehole", "sample", "HR", "bitumen"])
    general_data_df = general_data_df.drop_duplicates(subset=["project", "borehole", "sample"])
    df_dikes = pd.DataFrame(data=dike_table_data, columns=["dike_name", "waterboard", "notes"])

    master_table_df.to_csv(base_folder.joinpath("master_table.csv"), index=False)
    general_data_df.to_csv(base_folder.joinpath("general_data.csv"), index=False)
    df_dikes.to_csv(base_folder.joinpath("dike_table.csv"), index=False)

    toc = time.time()
    print(f"Time taken: {toc - tic:.2f} seconds")
