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
from src.processing.strength_processing import calc_linear_fit, correct_data, define_sec_modulus, \
    calc_fracture_data



KEYWORD_MAP = {
    "bezwijksterkte": "strength",
    "vermoeiing": "fatigue",
    "stijfheid": "stiffness",
}

# ----------------- HELPERS -----------------
def extract_num(s: str) -> int | None:
    """Extract first integer from string, or None if not found."""
    if m := re.search(r'\d+', s):
        return int(m.group())
    return None


def fill_master_table_data(project: int, vak_name: str, borehole_id_list: list[int], master_table_data: list, vak_data: pd.DataFrame) -> list:
    """
    The columns of the master table are: project, borehole, dijk, aanlegjaar, onderzoekjaar
    :param project:
    :param vak_name:
    :param borehole_id_list:
    :param master_table_data:
    :return:
    """
    construction_year =  vak_data['Aanleg-jaar'].iloc[0] if not vak_data.empty else None
    sample_year =  vak_data['onderzoeksjaar'].iloc[0] if not vak_data.empty else None

    for borehole_id in borehole_id_list:
        master_table_data.append(
            [f"P_{project}", f"BH{borehole_id}", f"{vak_name}", construction_year, sample_year]
        )
    return master_table_data


def fill_general_table_data(project_id: int, borehole_id, sample_name_strength: str, sample_name_fatigue: str,
                            sample_name_stiffness: str,
                            general_table_data: list, bh_data: pd.DataFrame) -> list:
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
    HR =  bh_data['HR'].iloc[0] if not bh_data.empty else None
    bitumen = bh_data['Bitumengehalte NEN'].iloc[0] if not bh_data.empty else None
    if sample_name_strength:
        general_table_data.append([f"P_{project_id}", f"BH{borehole_id}", sample_name_strength, HR, bitumen])
    if sample_name_fatigue:
        general_table_data.append([f"P_{project_id}", f"BH{borehole_id}", sample_name_fatigue, HR, bitumen])
    if sample_name_stiffness:
        general_table_data.append([f"P_{project_id}", f"BH{borehole_id}", sample_name_stiffness, HR, bitumen])
    return general_table_data


def fill_project_data_csv(base_folder: Path, project_names: list[int]):
    project_data = []
    for project_id in project_names:
        project_folder = base_folder.joinpath(f"P_{project_id}")
        project_folder.mkdir(exist_ok=True, parents=True)
        project_data.append({
            "project_name": f"P_{project_id}",
            "project_code": f"{project_id}",
            "notes": "AAAA"
        })

    df_projects = pd.DataFrame(data=project_data)
    df_projects.to_csv(base_folder.joinpath("project_table.csv"), index=False)


def fill_dike_data_table_df(vak_dict: dict, dike_data, project_master_table: pd.DataFrame):
    dike_names = list(vak_dict.keys())
    waterboard = project_master_table['Waterschap / beheerder'].dropna().unique().tolist()[0]
    for dike_name in dike_names:
        dike_data.append({
            "dike_name": dike_name,
            "waterboard": waterboard,
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


def fill_sample_data_csv(borehole_path: Path, sample_data: dict):

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
        r'^(ORG|Invoer|Resultaten|Grafieken|Org|Vermoeiingslijn|Layout voor rapportage|Resultaat|Blad1|Blad2)$', sample)]

    # remove all strings without a digit:
    samples = [s for s in samples if re.search(r'\d+', s)]

    # remove all strings which contains Blad:
    samples = [s for s in samples if 'Blad' not in s]
    return samples



def load_samples(file_path: Path) -> list[str]:
    """Wrapper for sheet-reading function with existence check."""
    if not file_path or not file_path.exists():
        return []
    return get_sample_names_from_sheet(file_path)

def assign_files_to_vak(vak_dict_mapping, project_master_table: pd.DataFrame, filename, input_files_folder):
    """Assign strength/fatigue/stiffness files to each vak based on filename keywords."""
    vak_list = project_master_table['Dijknaam'].dropna().unique().tolist()

    for vak in vak_list:
        vak_dict_mapping.setdefault(vak, {})
        vak_data = project_master_table.loc[project_master_table['Dijknaam'] == vak, 'boorkern_id']
        vak_dict_mapping[vak]["BH_ids"] = [int(bh_id) for bh_id in vak_data.dropna().unique().tolist()]

        for keyword, attr in KEYWORD_MAP.items():
            if keyword in filename.lower():
                fname = filename + ".xlsm"
                samples = load_samples(input_files_folder / fname)

                for s in samples:
                    if (num := extract_num(s)) and num in vak_dict_mapping[vak]["BH_ids"]:
                        vak_dict_mapping[vak][attr] = fname
                        break  # stop once a match is found
                break  # matched keyword, no need to check others


def process_borehole(project_id, bh_data: pd.DataFrame, borehole_id, base_folder, sample_names_mapping_dict, vak_files):
    """Process individual borehole: create dirs, export CSVs."""
    borehole_name = f"BH{borehole_id}"
    borehole_path = base_folder / f"P_{project_id}" / borehole_name
    borehole_path.mkdir(exist_ok=True, parents=True)


    strength_file = input_files_folder / vak_files.get("strength") if "strength" in vak_files else None
    fatigue_file = input_files_folder / vak_files.get("fatigue") if "fatigue" in vak_files else None
    stiffness_file = input_files_folder / vak_files.get("stiffness") if "stiffness" in vak_files else None

    if len(sample_names_mapping_dict[borehole_id]["fatigue"]) > 1:
        raise ValueError(
            f"Sample {sample_names_mapping_dict[borehole_id]['fatigue']} has multiple fatigue samples "
            f"(project {project_id}). Please fix the file so only one sample per borehole exists."
        )

    strength_sample, fatigue_sample, stiffness_sample = None, None, None  # will be modified to actual names if files exist
    sample_data = {}
    if strength_file:
        strength_sample = sample_names_mapping_dict[borehole_id]["strength"][0]
        fill_strength_data_csv(borehole_path, strength_sample, strength_file)

        D, h, strength, v = read_parameters(strength_file, strength_sample)

        sample_data[strength_sample] = {
            "depth": 0,
            "thickness": D,
            "height": h,
            "strength": strength,
            "v": v,
            "notes": [
                "DDDDDD"
            ]
        }
    else:
        test_path = borehole_path / f"strength"
        test_path.mkdir(exist_ok=True, parents=True)

    if fatigue_file:
        fatigue_sample = sample_names_mapping_dict[borehole_id]["fatigue"][0]
        fill_fatigue_data_csv(borehole_path, fatigue_sample, fatigue_file)
        sample_data[fatigue_sample] =  {
            "depth": 0,
            "notes": [
                "DDDDDD"
            ]
        }
    else:
        test_path = borehole_path / f"fatigue"
        test_path.mkdir(exist_ok=True, parents=True)

    if stiffness_file:
        stiffness_sample = sample_names_mapping_dict[borehole_id]["stiffness"][0]
        fill_stiffness_data_csv(borehole_path, stiffness_sample, stiffness_file)
        sample_data[stiffness_sample]= {
            "depth": 0,
            "notes": [
                "DDDDDD"
            ]
        }
    else:
        test_path = borehole_path / f"stiffness"
        test_path.mkdir(exist_ok=True, parents=True)

    fill_borehole_data_csv(borehole_path, borehole_name)
    fill_sample_data_csv(borehole_path, sample_data)
    fill_general_table_data(project_id, borehole_id, strength_sample, fatigue_sample, stiffness_sample, general_table_data, bh_data)


if __name__ == "__main__":
    tic = time.time()

    SCRIPT_DIR = Path(__file__).parent

    # Input path to modify
    base_folder = Path(r'C:\Users\marloes.slokker\Infram BV\Infram Projecten - 23i740_KC WAB 2024\Uitvoering\output_script_levensduurmodel') # Path to store the formatted data structure
    input_files_folder = Path(
        r'C:\Users\marloes.slokker\Infram BV\Infram Projecten - 23i740_KC WAB 2024\Uitvoering\levensduurmodel WAB\all_projectnumbers_files')  # make the path a env variable
    input_general_data_file = Path(r"C:\Users\marloes.slokker\Infram BV\Infram Projecten - 23i740_KC WAB 2024\Uitvoering\levensduurmodel WAB\Database Asfalt Excel.xlsx")


    ## START
    if base_folder.is_dir():
        shutil.rmtree(base_folder)
    base_folder.mkdir(exist_ok=True, parents=True)

    input_general_data_table = pd.read_excel(input_general_data_file, sheet_name="Database", dtype={"Projectnummer": str,})
    input_general_data_table['Projectnummer'] = input_general_data_table['Projectnummer'].astype(str)
    projects_ids = input_general_data_table['Projectnummer'].dropna().unique().tolist()
    projects_ids.reverse()
    projects_ids = ['0803318'] # TODO : process only the projects in input_files_folder

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
        project_master_table = input_general_data_table[input_general_data_table['Projectnummer'] == project_id]

        vak_list = project_master_table['Dijknaam'].dropna().unique().tolist()


        for file in input_files_folder.glob("*.xlsm"):
            if project_id not in file.stem:
                continue
            assign_files_to_vak(vak_dict_mapping, project_master_table, file.stem, input_files_folder)



        fill_dike_data_table_df(vak_dict_mapping, dike_table_data, project_master_table)

        for vak_name, vak_files in vak_dict_mapping.items():

            if "strength" in vak_files:
                strength_file = input_files_folder.joinpath(vak_files.get("strength"))
                sample_name_strength = get_sample_names_from_sheet(strength_file)
            else:
                sample_name_strength = []

            if "fatigue" in vak_files:
                fatigue_file = input_files_folder.joinpath(vak_files.get("fatigue"))
                sample_name_fatigue = get_sample_names_from_sheet(fatigue_file)
            else:
                sample_name_fatigue = []

            if "stiffness" in vak_files:
                stiffness_file = input_files_folder.joinpath(vak_files.get("stiffness"))
                sample_name_stiffness = get_sample_names_from_sheet(stiffness_file)
            else:
                sample_name_stiffness = []


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

            # borehole_ids = list(sample_names_mapping_dict.keys())
            borehole_ids = vak_files["BH_ids"]
            borehole_ids.sort()
            vak_data = project_master_table.loc[project_master_table['Dijknaam'] == vak_name]

            fill_master_table_data(project_id, vak_name, borehole_ids, master_table_data, vak_data)
            # There can be one borehole without strength because the test was bad or something.

            for borehole_id in borehole_ids:
                bh_data = vak_data.loc[vak_data['boorkern_id'] == borehole_id]

                process_borehole(project_id, bh_data, borehole_id, base_folder, sample_names_mapping_dict, vak_files)

    pd.DataFrame(master_table_data, columns=["project", "borehole", "dijk", "aanlegjaar", "onderzoeksjaar"]) \
        .to_csv(base_folder / "master_table.csv", index=False)

    pd.DataFrame(general_table_data, columns=["project", "borehole", "sample", "HR", "bitumen"]) \
        .drop_duplicates(subset=["project", "borehole", "sample"]) \
        .to_csv(base_folder / "general_data.csv", index=False)

    pd.DataFrame(dike_table_data, columns=["dike_name", "waterboard", "notes"]) \
        .to_csv(base_folder / "dike_table.csv", index=False)

    toc = time.time()
    print(f"Time taken: {toc - tic:.2f} seconds")
