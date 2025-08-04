import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import webbrowser


if __name__ == "__main__":

    # data_path = os.environ["DATA_PATH"]
    SCRIPT_DIR = Path(__file__).parent
    data_path = SCRIPT_DIR.parent.parent / "data"
    data_file = data_path / "from_bernadette/Database WAB - overzicht ADL28062023_BWich_selection18.8.xlsx"
    result_path = Path("results")
    result_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(data_file)

    columns = {
        "Dijknaam": "dijk",
        "Projectnummer": "project",
        "leeftijd": "age",
        "HR": "void_ratio",
        "Bitumen-gehalte NEN": "bitumen",
        "Buigtreksterkte": "str"
    }
    df = df[list(columns.keys())]
    df = df.rename(columns=columns)
    df["bitumen"] = pd.to_numeric(df["bitumen"], errors="coerce")
    df["dummy"] = "Total"  # Hack for getting the total summary easily

    # Regression features for "new formula"
    df["feat_1"] = np.where(df["age"]<=40, df["age"]**2, df["void_ratio"])
    df["feat_2"] = np.where(df["age"]<=40, df["age"]**3, df["void_ratio"]**3)
    df["feat_3"] = np.where(df["age"]<=40, df["void_ratio"], df["void_ratio"]**2*df["age"]**2)

    # pairplot = sns.pairplot(data=df, vars=["age", "void_ratio", "bitumen", "str"], hue="dijk")
    # pairplot.figure.savefig(result_path/"pairplot.png")

    pairplot = sns.pairplot(data=df, vars=["feat_1", "feat_2", "feat_3", "str"], hue="project")
    pairplot.figure.savefig(result_path/"pairplot.png")
