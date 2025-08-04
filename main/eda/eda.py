import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import webbrowser


if __name__ == "__main__":

    data_path = os.environ["DATA_PATH"]
    data_path = Path(data_path)
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

    dijk_summary_df = df.groupby("dijk").agg({
        "project": [("unique_count", lambda g: pd.unique(g).size)],
        "age": ["mean", "std", "min", "max", ("nan_count", lambda g: g.isna().sum())],
        "void_ratio": ["mean", "std", "min", "max", ("nan_count", lambda g: g.isna().sum())],
        "bitumen": ["mean", "std", "min", "max", ("nan_count", lambda g: g.isna().sum())],
        "str": ["mean", "std", "min", "max", ("nan_count", lambda g: g.isna().sum())],
    })

    total_summary_df = df.groupby("dummy").agg({
        "project": ["count"],
        "age": ["mean", "std", "min", "max", ("nan_count", lambda g: g.isna().sum())],
        "void_ratio": ["mean", "std", "min", "max", ("nan_count", lambda g: g.isna().sum())],
        "bitumen": ["mean", "std", "min", "max", ("nan_count", lambda g: g.isna().sum())],
        "str": ["mean", "std", "min", "max", ("nan_count", lambda g: g.isna().sum())],
    })

    summary_df = pd.concat([dijk_summary_df, total_summary_df])

    summary_df.to_csv(result_path/"summary.csv", index=False)

    styled_summary_df = summary_df.style \
        .format(precision=2) \
        .background_gradient(cmap='Blues', axis=0) \
        .set_caption("Dijk Summary Statistics") \
        .highlight_null(null_color='red') \
        .set_table_styles([{
        'selector': 'caption',
        'props': [('color', 'black'), ('font-size', '16px')]
    }])
    styled_summary_df.to_html(result_path/"summary_table.html")
    webbrowser.open(result_path/"summary_table.html")



