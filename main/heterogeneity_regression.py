import matplotlib.pyplot as plt
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

    df["age_at_investigation"] = df["investigation_year"] - df["construction_year"]

    return df


def save_results(res: Dict[str, Any], df: pd.DataFrame, path: Path, log_y: bool = True) -> None:

    res_lists = {}
    for (key_1, val_1) in res.items():
        if isinstance(val_1, dict):
            val_1 = {key_2: val_2.tolist() if isinstance(val_2, np.ndarray) else val_2 for (key_2, val_2) in val_1.items()}
        else:
            val_1 = val_1.tolist() if isinstance(val_1, np.ndarray) else val_1
        res_lists[key_1] = val_1

    with open(path, "w") as f:
        json.dump(res_lists, f, indent=4)

    #add values to df and output as csv for further analysis
    if log_y:
        df['log_sig_b_detrended'] = res['Homogeen']['resid_all']
        df['log_sig_b_regression'] = res['Homogeen']['fitted_values_all']
    else:
        df['sig_b_detrended'] = res['Homogeen']['resid_all']
        df['sig_b_regression'] = res['Homogeen']['fitted_values_all']
    df.to_csv(path.parent / "data_with_regression_output.csv", index=False)

def fit_linear_regression(df: pd.DataFrame, heterogeneity_category: str = "all") -> Dict[str, Any]:

    if heterogeneity_category != "all":
        df_training = df.loc[df["heterogeneity_category"] == heterogeneity_category].reset_index(drop=True)
    else:
        df_training = df.copy()

    X = df_training["age_at_investigation"]
    X = sm.add_constant(X)
    y = df_training["target"]

    model = sm.OLS(y, X).fit()
    summary = model.summary()

    fitted_values_all = model.predict(sm.add_constant(df["age_at_investigation"])).values

    results = {
        "heterogeneity_category": heterogeneity_category,
        "beta": model.params.values,
        "se": np.sqrt(model.scale),
        "beta_ste": model.bse.values,
        "beta_cov": model.cov_HC0,
        "t_values": model.tvalues.values,
        "p_values": model.pvalues.values,
        "conf_int": model.conf_int().values,
        "r_2": model.rsquared,
        "X_training": df_training["age_at_investigation"].values,
        "X_all": df["age_at_investigation"].values,
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


def fit_adjusted_linear_regression(df: pd.DataFrame, lr_results: Dict[str, Any]) -> Dict[str, Any]:

    df_heterogenous = df.loc[df["heterogeneity_category"] == "Heterogeen"].reset_index(drop=True)

    beta_homogenous = lr_results["beta"]
    beta_ste_homogenous = lr_results["beta_ste"]

    df_heterogenous["target"] -= beta_homogenous[0] + beta_homogenous[1] * df_heterogenous["age_at_investigation"]

    res = fit_linear_regression(df_heterogenous, heterogeneity_category="Heterogeen")

    return res


def plot_lr_fit(lr_results: Dict[str, Dict[str, Any]], path: Path, key: str, log_y: bool = True) -> None:

    data = lr_results[key]
    X_training = data["X_training"]
    y_training = data["target_training"]

    X_all = data["X_all"]
    idx_sort = np.argsort(X_all)
    X_all = X_all[idx_sort]
    y_all = data["target_all"][idx_sort]
    y_hat = data["fitted_values_all"][idx_sort]
    se = data["se"]

    if log_y:

        fig, axs = plt.subplots(1, 2, sharex=True, figsize=(12, 6))

        ax = axs[0]
        ax.scatter(X_all, y_all, c="b", alpha=0.4, label="Entire dataset")
        ax.scatter(X_training, y_training, c="r", alpha=0.4, label=f"Training ({key.lower()}) dataset")
        ax.plot(X_all, y_hat, c="r", label="Regression fit")
        ax.set_xlabel("Age [yr]", fontsize=14)
        ax.set_ylabel("$ln({σ}_{b})$ [ln(kPa)]", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid()
        ax.set_title("Age-$ln({σ}_{b})$ regression")

        ax = axs[1]
        ax.scatter(X_all, np.exp(y_all), c="b", alpha=0.4, label="Entire dataset")
        ax.scatter(X_training, np.exp(y_training), c="r", alpha=0.4, label=f"Training ({key.lower()}) dataset")
        ax.plot(X_all, np.exp(y_hat+0.5*se**2), c="r", label="Regression fit")
        ax.set_xlabel("Age [yr]", fontsize=14)
        ax.set_ylabel("${σ}_{b}$ [kPa]", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid()
        ax.set_title("Age-${σ}_{b}$ transformation")

    else:

        fig = plt.figure()
        plt.scatter(X_all, y_all, c="b", alpha=0.4, label="Entire dataset")
        plt.scatter(X_training, y_training, c="r", alpha=0.4, label=f"Training ({key.lower()}) dataset")
        plt.plot(X_all, y_hat, c="r", label="Regression fit")
        plt.xlabel("Age [yr]", fontsize=14)
        plt.ylabel("${σ}_{b}$ [kPa]", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid()

    plt.close()
    fig.savefig(path/f"linear_regression_{key.lower()}.png")

    pass


def plot_fits(lr_results: Dict[str, Dict[str, Any]], path: Path, log_y: bool = True) -> None:

    path = path / "plots"
    path.mkdir(parents=True, exist_ok=True)

    plot_lr_fit(lr_results, path, key="Homogeen", log_y=log_y)

    pass





def main(log_y: bool = True) -> None:

    script_path = Path(__file__).parent
    # data_path = script_path.parent / "data/db_querry.csv"
    data_path = script_path.parent / "data/database_all_v3.csv"
    result_path = script_path.parent / f"results/heterogeneity_regression/log_y_{log_y}"
    result_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    df = set_heterogeneity(df)
    # df = df.loc[df['age']>30]
    df = df.loc[df["heterogeneity_category"].isin(["Homogeen", "Heterogeen", "Matig heterogeen"])].reset_index(drop=True)
    #drop rows where sig_b is nan
    df = df.dropna(subset=["sig_b"]).reset_index(drop=True)
    if log_y:
        df["target"] = np.log(df["sig_b"])
    else:
        df["target"] = df["sig_b"]

    lr_results = {
        category: fit_linear_regression(df, heterogeneity_category=category)
        for category in pd.unique(df["heterogeneity_category"])
    }

    lr_results["Heterogeen_adjusted"] = fit_adjusted_linear_regression(df, lr_results["Homogeen"])

    save_results(lr_results, df, result_path/"lr_results.json",log_y=log_y)

    plot_fits(lr_results, result_path, log_y=log_y)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--log_y", action="store_false")
    args = parser.parse_args()
    log_y = True
    main(
        log_y= log_y
    )

