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
    data_file = data_path / "from_bernadette/bronze/Database WAB - overzicht ADL28062023_BWich_selection18.8.xlsx"
    result_path = Path(SCRIPT_DIR.parent.parent/"results/eda")
    result_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(data_file)

    columns = {
        "Dijknaam": "dijk",
        "Aanleg-jaar": "const_year",
        "onderzoeksjaar": "meas_year",
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
    df["const_year"] = pd.to_numeric(df["const_year"], errors="coerce")
    df["meas_year"] = pd.to_numeric(df["meas_year"], errors="coerce")

    # Regression features for "new formula"
    df["feat_1"] = np.where(df["age"]<=40, df["age"]**2, df["void_ratio"])
    df["feat_2"] = np.where(df["age"]<=40, df["age"]**3, df["void_ratio"]**3)
    df["feat_3"] = np.where(df["age"]<=40, df["void_ratio"], df["void_ratio"]**2*df["age"]**2)
    df["str/age"] = df["str"] / (df["age"] + 1)

    fig = plt.figure()
    # pairplot = sns.pairplot(data=df, vars=["age", "void_ratio", "bitumen", "str"], hue="dijk")
    pairplot = sns.pairplot(data=df, vars=["feat_1", "feat_2", "feat_3", "str"], hue="project")
    pairplot.figure.savefig(result_path/"pairplot.png")

    fig = plt.figure()
    yearplot = sns.scatterplot(data=df.loc[(df["age"] >= 10) &(df["age"] <= 30)], x="const_year", y="str", hue="age")
    # yearplot = sns.scatterplot(data=df, x="const_year", y="str/age")
    plt.yscale("log")
    yearplot.figure.savefig(result_path/"yearplot.png")

    measyear_grid = np.linspace(2000, 2025, 10)
    constyear_grid = np.linspace(1950, 2025, 10)
    measyear_mesh, constyear_mesh = np.meshgrid(measyear_grid, constyear_grid)
    diff_mesh = measyear_mesh - constyear_mesh

    fig = plt.figure()
    cs = plt.contour(constyear_mesh, measyear_mesh, diff_mesh, levels=list(range(0, 100, 10)), colors="k")
    plt.clabel(cs, inline=True, fontsize=10, fmt="Age=%dy")
    for dike in pd.unique(df["dijk"]):
        data = df.loc[df["dijk"] == dike]
        x = data["const_year"].values
        y = data["meas_year"].values
        age = data["age"].values
        idx = np.argsort(age)
        x = x[idx]
        y = y[idx]
        plt.plot(x, y)
    df["str_av"] = df.groupby("dijk")["str"].transform("mean")
    sc = plt.scatter(df["const_year"], df["meas_year"], c=df["str_av"], cmap="inferno")
    cbar = plt.colorbar(sc, label="Average strength of boreholes [kPa]")
    plt.xlabel("Construction year")
    plt.ylabel("Measurement year")
    fig.savefig(result_path/"yearcontourplot_avgstr.png")


    fig = plt.figure()
    cs = plt.contour(constyear_mesh, measyear_mesh, diff_mesh, levels=list(range(0, 100, 10)), colors="k")
    plt.clabel(cs, inline=True, fontsize=10, fmt="Age=%dy")
    for dike in pd.unique(df["dijk"]):
        data = df.loc[df["dijk"] == dike]
        x = data["const_year"].values
        y = data["meas_year"].values
        age = data["age"].values
        idx = np.argsort(age)
        x = x[idx]
        y = y[idx]
        plt.plot(x, y)
    df["str_av"] = df.groupby("dijk")["str"].transform("mean")
    df["str_std"] = df.groupby("dijk")["str"].transform("std")
    df["str_cov"] = df["str_std"] / df["str_av"]
    sc = plt.scatter(df["const_year"], df["meas_year"], c=df["str_cov"], cmap="inferno")
    cbar = plt.colorbar(sc, label="CoV of boreholes [kPa]")
    plt.xlabel("Construction year")
    plt.ylabel("Measurement year")
    fig.savefig(result_path/"yearcontourplot_covstr.png")
