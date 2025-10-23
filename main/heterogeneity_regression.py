import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
import json
from argparse import ArgumentParser
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Optional, Any


def categorize_heterogeneity(cov: float) -> str:
    if cov < 0.2:
        return 'Homogeen'
    elif 0.2 <= cov < 0.35:
        return 'Matig heterogeen'
    elif cov >= 0.35:
        return 'Heterogeen'


def set_heterogeneity(df: pd.DataFrame) -> pd.DataFrame:

    df_dike_project = pd.DataFrame.groupby(df, by="project_dike_id").agg({"sig_b": ["mean", "std"]})

    df_dike_project = df_dike_project.dropna(how="any")
    df_dike_project.columns = ['_'.join(col).strip() for col in df_dike_project.columns.values]
    df_dike_project.loc[:, "sig_b_cov"] = df_dike_project["sig_b_std"] / df_dike_project["sig_b_mean"]
    df_dike_project["project_dike_id"] = df_dike_project.index
    df_dike_project = df_dike_project.reset_index(drop=True)

    df = df.merge(
        df_dike_project.loc[:, ["project_dike_id", "sig_b_cov"]],
        on="project_dike_id",
        how="left"
    )

    df['heterogeneity_category'] = df['sig_b_cov'].apply(categorize_heterogeneity)

    df["age"] = df["investigation_year"] - df["construction_year"]

    return df


def fit_linear_regression(df: pd.DataFrame, heterogeneity_category: str = "all") -> Dict[str, Any]:

    if heterogeneity_category != "all":
        df_training = df.loc[df["heterogeneity_category"] == heterogeneity_category].copy()
    else:
        df_training = df.copy()

    X = df_training["age"]
    X = sm.add_constant(X)
    y = df_training["target"]

    model = sm.OLS(y, X).fit()
    summary = model.summary()

    fitted_values_all = model.predict(sm.add_constant(df["age"])).values

    results = {
        "heterogeneity_category": heterogeneity_category,
        "beta": model.params.values,
        "ste": np.sqrt(model.scale),
        "beta_std": model.bse.values,
        "beta_cov": model.cov_HC0,
        "t_values": model.tvalues.values,
        "p_values": model.pvalues.values,
        "conf_int": model.conf_int().values,
        "r_2": model.rsquared,
        "X_training": df_training["age"].values,
        "X_all": df["age"].values,
        "target_training": df_training["target"].values,
        "target_all": df["target"].values,
        "fitted_values_training": model.fittedvalues.values,
        "resid_training": model.resid.values,
        "fitted_values_all": fitted_values_all,
        "resid_all": df["target"].values - fitted_values_all,
        "n_obs": int(model.nobs),
        "summary": summary.as_text(),
        "summary_dict": model.summary2().tables[1].to_dict(orient='index'),
    }

    return results

def save_results(res: Dict[str, Any], path) -> None:

    for (key_1, val_1) in res.items():
        if isinstance(val_1, dict):
            val_1 = {key_2: val_2.tolist() if isinstance(val_2, np.ndarray) else val_2 for (key_2, val_2) in val_1.items()}
        else:
            val_1 = val_1.tolist() if isinstance(val_1, np.ndarray) else val_1
        res[key_1] = val_1

    with open(path, "w") as f:
        json.dump(res, f, indent=4)


def main(log_y: bool = True) -> None:

    script_path = Path(__file__).parent
    data_path = script_path.parent / "data/db_querry.csv"
    result_path = script_path.parent / f"results/heterogeneity_regression/log_y_{log_y}"
    result_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    df = set_heterogeneity(df)

    df = df.loc[df["heterogeneity_category"].isin(["Homogeen", "Heterogeen"])]

    if log_y:
        df["target"] = np.log(df["sig_b"])
    else:
        df["target"] = df["sig_b"]

    lr_results = {category: fit_linear_regression(df, heterogeneity_category=category) for category in pd.unique(df["heterogeneity_category"])}
    save_results(lr_results, result_path/"lr_results.json")




    pass


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--log_y", action="store_false")
    args = parser.parse_args()

    main(
        log_y=args.log_y
    )

