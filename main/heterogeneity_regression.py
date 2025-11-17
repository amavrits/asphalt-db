import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import json
from argparse import ArgumentParser
from src.old_fit import *
from typing import List, Tuple, Dict, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style("whitegrid")


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


def fit_linear_regression(
        df: pd.DataFrame,
        heterogeneity_category: str = "all",
        regress_HR: bool = False
) -> Dict[str, Any]:

    if heterogeneity_category != "all":
        df_training = df.loc[df["heterogeneity_category"] == heterogeneity_category].reset_index(drop=True)
    else:
        df_training = df.copy()

    if regress_HR:
        X = df_training[["age_at_investigation", "HR"]]
    else:
        X = df_training["age_at_investigation"]
    X = sm.add_constant(X)
    y = df_training["target"]

    model = sm.OLS(y, X).fit()
    summary = model.summary()

    if regress_HR:
        prediction_training = model.get_prediction(sm.add_constant(df_training[["age_at_investigation", "HR"]]))
    else:
        prediction_training = model.get_prediction(sm.add_constant(df_training["age_at_investigation"]))
    summary_prediction_training = prediction_training.summary_frame(alpha=0.10)
    mean_prediction_training = summary_prediction_training["mean"].values
    ci_prediction_training = summary_prediction_training[["mean_ci_lower", "mean_ci_upper"]].values
    pi_prediction_training = summary_prediction_training[["obs_ci_lower", "obs_ci_upper"]].values

    if regress_HR:
        prediction_all = model.get_prediction(sm.add_constant(df[["age_at_investigation", "HR"]]))
    else:
        prediction_all = model.get_prediction(sm.add_constant(df["age_at_investigation"]))
    summary_prediction_all = prediction_all.summary_frame(alpha=0.10)
    mean_prediction_all = summary_prediction_all["mean"].values
    ci_prediction_all = summary_prediction_all[["mean_ci_lower", "mean_ci_upper"]].values
    pi_prediction_all = summary_prediction_all[["obs_ci_lower", "obs_ci_upper"]].values

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
        "X_training": df_training[["age_at_investigation", "HR"]].values if regress_HR else df_training["age_at_investigation"].values,
        "X_all": df[["age_at_investigation", "HR"]].values if regress_HR else df["age_at_investigation"].values,
        "target_training": df_training["target"].values,
        "target_all": df["target"].values,
        "fitted_values_training": mean_prediction_training,
        "ci_prediction_training": ci_prediction_training,
        "pi_prediction_training": pi_prediction_training,
        "resid_training": model.resid.values,
        "fitted_values_all": mean_prediction_all,
        "ci_prediction_all": ci_prediction_all,
        "pi_prediction_all": pi_prediction_all,
        "resid_all": df["target"].values - mean_prediction_all,
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
    X_training = X_training if X_training.ndim == 1 else X_training[:, 0]
    y_training = data["target_training"]

    X_all = data["X_all"]
    X_all = X_all if X_all.ndim == 1 else X_all[:, 0]
    idx_sort = np.argsort(X_all)
    X_all = X_all[idx_sort]
    y_all = data["target_all"][idx_sort]
    y_hat = data["fitted_values_all"][idx_sort]
    ci = data["ci_prediction_all"][idx_sort]
    pi = data["pi_prediction_all"][idx_sort]
    se = data["se"]

    if log_y:

        fig, axs = plt.subplots(1, 2, sharex=True, figsize=(16, 6))

        ax = axs[0]
        ax.fill_between(X_all, pi[:, 0], pi[:, 1], color="r", alpha=0.1, label="95% PI")
        ax.fill_between(X_all, ci[:, 0], ci[:, 1], color="r", alpha=0.3, label="95% CI")
        ax.scatter(X_all, y_all, c="b", alpha=0.4, label="Volledige dataset")
        ax.scatter(X_training, y_training, c="r", alpha=0.4, label=f"Training ({key.lower()}) dataset")
        ax.plot(X_all, y_hat, c="r", label="Regression fit")
        ax.set_xlabel("Leeftijd [jaar]", fontsize=14)
        ax.set_ylabel("$ln({σ}_{b})$ [ln(kPa)]", fontsize=14)
        ax.legend(fontsize=12, loc="lower left")
        ax.grid()
        ax.set_title("Leeftijd-$ln({σ}_{b})$")

        ax = axs[1]
        ax.fill_between(X_all, np.exp(pi[:, 0]), np.exp(pi[:, 1]), color="r", alpha=0.1, label="95% PI")
        ax.fill_between(X_all, np.exp(ci[:, 0]+0.5*se**2), np.exp(ci[:, 1]+0.5*se**2), color="r", alpha=0.3, label="95% CI")
        ax.scatter(X_all, np.exp(y_all+0.5*se**2), c="b", alpha=0.4, label="Volledige dataset")
        ax.scatter(X_training, np.exp(y_training), c="r", alpha=0.4, label=f"Training ({key.lower()}) dataset")
        ax.plot(X_all, np.exp(y_hat+0.5*se**2), c="r", label="Regression fit")
        ax.set_xlabel("Leeftijd [jaar]", fontsize=14)
        ax.set_ylabel("${σ}_{b}$ [kPa]", fontsize=14)
        ax.legend(fontsize=12, loc="lower left")
        ax.grid()
        ax.set_title("Leeftijd-${σ}_{b}$")

    else:

        fig = plt.figure()
        plt.fill_between(X_all, pi[:, 0], pi[:, 1], color="r", alpha=0.1, label="95% PI")
        plt.fill_between(X_all, ci[:, 0], ci[:, 1], color="r", alpha=0.3, label="95% CI")
        plt.scatter(X_all, y_all, c="b", alpha=0.4, label="Volledige dataset")
        plt.scatter(X_training, y_training, c="r", alpha=0.4, label=f"Training ({key.lower()}) dataset")
        plt.plot(X_all, y_hat, c="r", label="Regression fit")
        plt.xlabel("Leeftijd [jaar]", fontsize=14)
        plt.ylabel("${σ}_{b}$ [kPa]", fontsize=14)
        plt.legend(fontsize=12, loc="lower left")
        plt.grid()

    fig.suptitle("${R}^{2}$="+f"{data['r_2']*100:.0f}%")
    fig.savefig(path/f"linear_regression_{key.lower()}.png")
    plt.close()


def plot_fits(lr_results: Dict[str, Dict[str, Any]], path: Path, log_y: bool = True) -> None:

    path = path / "plots"
    path.mkdir(parents=True, exist_ok=True)

    plot_lr_fit(lr_results, path, key="Homogeen", log_y=log_y)


def main(log_y: bool = False, regress_HR: bool = False) -> None:

    script_path = Path(__file__).parent
    # data_path = script_path.parent / "data/database_all_v3.csv"
    data_path = script_path.parent / "data/database_all_v5.csv"
    result_path = script_path.parent / f"results/heterogeneity_regression/log_y_{log_y}"
    result_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    df = set_heterogeneity(df)
    df = df.loc[df["heterogeneity_category"].isin(["Homogeen", "Heterogeen", "Matig heterogeen"])].reset_index(drop=True)
    df = df.dropna(subset=["sig_b"]).reset_index(drop=True)
    if log_y:
        df["target"] = np.log(df["sig_b"])
    else:
        df["target"] = df["sig_b"]

    # df = df.loc[df["heterogeneity_category"] != "Matig heterogeen"]

    lr_results = {
        category: fit_linear_regression(df, heterogeneity_category=category, regress_HR=regress_HR)
        for category in pd.unique(df["heterogeneity_category"]).tolist()+["all"]
    }

    sns.scatterplot(df, x="age_at_investigation", y="sig_b", hue="heterogeneity_category")
    plt.show()

    fig = plt.figure(figsize=(10, 4))
    colors = ["r", "b"]
    for i, (key, val) in enumerate(lr_results.items()):
        if key == "all" or key == "Matig heterogeen":
            continue
        plt.scatter([val["beta"][-1]], [key], c=colors[i])
        plt.plot([val["beta"][-1]-1.96*val["beta_ste"][-1], val["beta"][-1]+1.96*val["beta_ste"][-1]], [key, key], c=colors[i])
    plt.xlabel("Leeftijd coefficient [kPa/jr]")
    plt.ylabel("Heterogeniteit categorie")
    plt.show()


    df["yhat_M1"] = lr_results["Homogeen"]["fitted_values_all"]
    df["resid_M1"] = df["target"] - df["yhat_M1"]
    df["pi_lower_M1"] = lr_results["Homogeen"]["pi_prediction_all"][:, 0]
    df["pi_upper_M1"] = lr_results["Homogeen"]["pi_prediction_all"][:, 1]

    df["HR_feat"] = (df["HR"].max() - df["HR"]) ** 2
    df["target_M2"] = np.log1p(-df["resid_M1"].min()+df["resid_M1"])

    df_M2 = df.loc[df["heterogeneity_category"] != "Homogeen"]
    model = sm.OLS(df_M2["target_M2"], sm.add_constant(df_M2["HR_feat"])).fit()
    prediction = model.get_prediction(sm.add_constant(df_M2["HR_feat"]))
    summary_prediction = prediction.summary_frame(alpha=0.10)
    mean_prediction = summary_prediction["mean"].values
    ci_prediction = summary_prediction[["mean_ci_lower", "mean_ci_upper"]].values
    pi_prediction = summary_prediction[["obs_ci_lower", "obs_ci_upper"]].values
    df_M2["yhat_M2"] = mean_prediction
    df_M2["pi_lower_M2"] = pi_prediction[:, 0]
    df_M2["pi_upper_M2"] = pi_prediction[:, 1]

    df = pd.concat((df.loc[df["heterogeneity_category"] == "Homogeen"], df_M2), axis=0)
    df = df.sort_values(by=["age_at_investigation"]).reset_index(drop=True)

    df["y_hat"] = np.where(
        df["heterogeneity_category"] == "Homogeen",
        df["yhat_M1"],
        df["yhat_M1"] + np.exp(df["yhat_M2"]+0.5*model.scale) - 1 + df["resid_M1"].min()
    )
    
    df["pi_lower"] = np.where(
        df["heterogeneity_category"] == "Homogeen",
        df["pi_lower_M1"],
        df["pi_lower_M1"] + np.exp(df["pi_lower_M2"]) - 1 + df["resid_M1"].min()
    )
    
    df["pi_upper"] = np.where(
        df["heterogeneity_category"] == "Homogeen",
        df["pi_upper_M1"],
        df["pi_upper_M1"] + np.exp(df["pi_upper_M2"]) - 1 + df["resid_M1"].min()
    )
    
    r2_hom = r2_score(df.loc[df["heterogeneity_category"]=="Homogeen", "target"], df.loc[df["heterogeneity_category"]=="Homogeen", "y_hat"])
    r2_het = r2_score(df.loc[df["heterogeneity_category"]=="Heterogeen", "target"], df.loc[df["heterogeneity_category"]=="Heterogeen", "y_hat"])
    r2_all = r2_score(df["target"], df["y_hat"])

    fig = plt.figure()
    sns.scatterplot(data=df, x="target", y="y_hat", hue="heterogeneity_category")
    plt.axline([0, 0], slope=1, c="k")
    plt.xlabel("${σ}_{b}$ Data [kPa]")
    plt.ylabel("${σ}_{b}$ Model [kPa]")
    plt.suptitle(f"Homogeen R^2={r2_hom*100:.0f}%\nHeterogeen R^2={r2_het*100:.0f}%\nTotal R^2={r2_all*100:.0f}%", fontsize=12)
    plt.show()

    fig = plt.figure()
    colors = {"Heterogeen": "b", "Matig heterogeen": "g", "Homogeen": "r"}
    for key in ["Homogeen", "Matig heterogeen", "Heterogeen"]:
        data = df.loc[df["heterogeneity_category"]==key]
        plt.scatter(data["age_at_investigation"], data["sig_b"], c=colors[key], marker="x", label=key)
    plt.plot(df["age_at_investigation"], df["yhat_M1"], c=colors["Homogeen"])
    plt.fill_between(df["age_at_investigation"], df["pi_lower_M1"], df["pi_upper_M1"], color="r", alpha=0.3)
    plt.plot()
    plt.xlabel("Leeftijd [jr]")
    plt.ylabel("${σ}_{b}$ [kPa]")
    plt.legend()
    plt.show()

    fig = plt.figure()
    colors = {"Heterogeen": "b", "Matig heterogeen": "g", "Homogeen": "r"}
    for key in ["Homogeen", "Matig heterogeen", "Heterogeen"]:
        data = df.loc[df["heterogeneity_category"]==key]
        y_err = np.array([data["y_hat"] - data["pi_lower"], data["pi_upper"] - data["y_hat"]])
        plt.errorbar(
            x=data["age_at_investigation"],
            y=data["y_hat"],
            yerr=y_err,
            c=colors[key], fmt="o", capsize=5, alpha=0.5, label=key
        )
        plt.scatter(data["age_at_investigation"], data["sig_b"], c=colors[key], marker="x")
    plt.xlabel("Leeftijd [jr]")
    plt.ylabel("${σ}_{b}$ [kPa]")
    plt.legend()
    plt.show()

    fig = plt.figure()
    plt.scatter(df_M2["age_at_investigation"], df_M2["resid_M1"])
    plt.xlabel("Leeftijd [jr]")
    plt.ylabel("Detrended heterogeen data")
    plt.show()

    fig = plt.figure()
    plt.scatter(df_M2["HR"], df_M2["resid_M1"])
    plt.xlabel("HR [%]")
    plt.ylabel("Detrended heterogeen data")
    plt.show()

    fig = plt.figure()
    plt.scatter(df_M2["HR_feat"], df_M2["target_M2"])
    plt.xlabel("${(max(HR)-HR)}^{2}$")
    plt.ylabel("ln(1-min(r)+r)\nr is the vector of detrended residuals")
    plt.show()

    old_model_mean, old_model_pi = fit_old_formula(df)
    df["old_model_mean_HR_4%"] = old_model_mean
    df["pi_lower_old_HR_4%"] = old_model_pi[:, 0]
    df["pi_upper_old_HR_4%"] = old_model_pi[:, 1]

    fig = plt.figure()
    plt.scatter(df["age_at_investigation"], df["sig_b"], c="b")
    plt.plot(df["age_at_investigation"], df["old_model_mean_HR_4%"], c="r")
    plt.fill_between(df["age_at_investigation"], df["pi_lower_old_HR_4%"], df["pi_upper_old_HR_4%"], color="r", alpha=0.3)
    plt.xlabel("Leeftijd [jr]")
    plt.ylabel("${σ}_{b}$ [kPa]")
    plt.show()


    df.to_csv(result_path/"data_with_regression_output.csv")

    plot_fits(lr_results, result_path, log_y)

    for (key, val) in lr_results.items():
        for (key2, val2) in val.items():
            lr_results[key][key2] = val2.tolist() if isinstance(val2, np.ndarray) else val2

    with open(result_path/"lr_results_M1.json", "w") as f:
        json.dump(lr_results, f, indent=4)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--log_y", action="store_true")
    parser.add_argument("--regress_HR", action="store_true")
    args = parser.parse_args()

    main(
        log_y=args.log_y,
        regress_HR=args.regress_HR
    )

