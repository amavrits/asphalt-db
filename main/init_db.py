import os
import pandas as pd
from src.config import DB_CONFIG
from src.db_builder.models import *
from src.db_builder.build import *
from src.db_builder.utils import *
from pathlib import Path
from dotenv import load_dotenv
import json


if __name__ == "__main__":

    # -----------------------
    # Load environment variables
    # -----------------------

    SCRIPT_DIR = Path(__file__).parent
    data_path = SCRIPT_DIR.parent / "data/dummy"

    dike_table, project_table, master_table, general_data = parse_base_data(data_path)

    create_db(DB_CONFIG)

    db.connect()

    create_tables(db)

    for project_folder in data_path.iterdir():

        if project_folder.is_file():
            continue

        project_name = project_folder.stem

        project_data = project_table.loc[project_name, :]
        add_project(project_name, project_data)

        iter_dikes(project_name, master_table, dike_table)

        for borehole_folder in project_folder.iterdir():

            if borehole_folder.is_file():
                continue

            borehole_name = borehole_folder.stem

            with open(borehole_folder/"borehole_data.json", "r") as f:
                borehole_data = json.load(f)

            add_borehole(borehole_name, project_name, master_table, borehole_data)

            with open(borehole_folder / "sample_data.json", "r") as f:
                sample_data = json.load(f)

            for (sample_name, data) in sample_data.items():

                add_sample(sample_name, borehole_name, data)

                add_sample_general_data(sample_name, borehole_name, project_name, general_data)

                add_sample_test()

                test_name = f"T_{sample_name}"
                test_folder_list = [file for file in borehole_folder.iterdir() if file.name.split(".")[-1] != "json"]
                test_name_list = [file.stem for file in test_folder_list]
                has_str = not any((borehole_folder/"str").iterdir())
                has_ftg = not any((borehole_folder/"ftg").iterdir())
                has_stf = not any((borehole_folder/"stf").iterdir())

                Test(
                    sample=Sample(sample_name=sample_name),
                    test_name=test_name,
                    strength=has_str,
                    fatigue=has_ftg,
                    stiffness=has_stf,
                )

            for test_folder in test_folder_list:

                df = pd.read_csv(test_folder/"raw_data.csv", index_col="sample_name")

                with open(test_folder / "test_data.json", "r") as f:
                    test_data = json.load(f)

                for sample_name in df.index:

                    data = df.loc[sample_name]

                    if test_folder.stem == "str":

                        StrSampleRaw(
                            test=Test(test_name=test_name),
                            notes=data["notes"],
                            t=data["t"],
                            F=data["F"],
                            V_org=data["V_org"],
                        )

                    elif test_folder.stem == "ftg":

                        FtgSampleRaw(
                            test=Test(test_name=test_name),
                            notes=data["notes"],
                            N=data["N"],
                            maximum_stroke=data["MaximumStroke"],
                            minimum_stroke=data["MinimumStroke"],
                            peak_to_peak_stroke=data["PeakToPeakStroke"],
                            maximum_load=data["MaximumLoad"],
                            peak_to_peak_load=data["PeakToPeakLoad"],
                            in_phase_modulus=data["InPhaseModulus"],
                            out_phase_modulus=data["OutPhaseModulus"],
                        )

                    elif test_folder.stem == "stf":

                        EdynSampleRaw(
                            test=Test(test_name=test_name),
                            notes=data["notes"],
                            T=data["T"],
                            f=data["f"],
                            eps=data["eps"],
                            E_dyn=data["E_dyn"],
                            pha=data["pha"]
                        )

                    else:
                        raise ValueError(f"Unknown test type {test_folder.stem}")


                    pass






    db.close()

