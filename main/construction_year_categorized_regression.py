import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from main.heterogeneity_regression import fit_linear_regression, set_heterogeneity
from pathlib import Path
import json
from argparse import ArgumentParser
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style("whitegrid")
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    category=ValueWarning,
    message="omni_normtest is not valid with less than 8 observations"
)


def fit_linear_regression_group(
        df: pd.DataFrame,
        subset_group: str = "construction_category",
        group: int = 0,
        regress_HR: bool = False
) -> Dict[str, Any]:

    df_training = df.loc[df[subset_group] == group].reset_index(drop=True)

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
    summary_prediction_training = prediction_training.summary_frame(alpha=0.05)
    mean_prediction_training = summary_prediction_training["mean"].values
    ci_prediction_training = summary_prediction_training[["mean_ci_lower", "mean_ci_upper"]].values
    pi_prediction_training = summary_prediction_training[["obs_ci_lower", "obs_ci_upper"]].values

    if regress_HR:
        prediction_all = model.get_prediction(sm.add_constant(df[["age_at_investigation", "HR"]]))
    else:
        prediction_all = model.get_prediction(sm.add_constant(df["age_at_investigation"]))
    summary_prediction_all = prediction_all.summary_frame(alpha=0.05)
    mean_prediction_all = summary_prediction_all["mean"].values
    ci_prediction_all = summary_prediction_all[["mean_ci_lower", "mean_ci_upper"]].values
    pi_prediction_all = summary_prediction_all[["obs_ci_lower", "obs_ci_upper"]].values

    results = {
        "group": group,
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
        "resid_training": df_training["target"].values - mean_prediction_training,
        "fitted_values_all": mean_prediction_all,
        "ci_prediction_all": ci_prediction_all,
        "pi_prediction_all": pi_prediction_all,
        "resid_all": df["target"].values - mean_prediction_all,
        "n_obs": int(model.nobs),
        "summary": summary.as_text(),
        "summary_dict": model.summary2().tables[1].to_dict(orient='index'),
    }

    return results


def main(
        n_groups: int  = 5,
        bin_method: str = "standard",
        log_y: bool = False,
        regress_HR: bool = False,
        verbose: bool = False
) -> None:

    script_path = Path(__file__).parent
    # data_path = script_path.parent / "data/database_all_v3.csv"
    data_path = script_path.parent / "data/database_all_v5.csv"
    result_path = script_path.parent / f"results/construction_year_categorized_regression/log_y_{log_y}/"
    result_path.mkdir(parents=True, exist_ok=True)
    plot_path = result_path / f"plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if log_y:
        df["target"] = np.log(df["sig_b"])
    else:
        df["target"] = df["sig_b"]

    # subset_group = "construction_category"
    # year_edges = [1900, 1973, 1993, 1995, 2025]
    # year_bins = [f"{year_1}-{year_2}" for (year_1, year_2) in zip(year_edges[:-1], year_edges[1:])]
    # idx = []
    # for i, row in df.iterrows():
    #     construction_year = row["construction_year"]
    #     idx.append(max([i if year < construction_year else -np.inf for i, year in enumerate(year_edges)]))
    # df[subset_group] = [year_bins[i] for i in idx]

    subset_group = "construction_category"
    df[subset_group] = 0
    year_ranges = [[1973, 1993], [1983, 2003], [1993, 2013], [2003, 2025]]

    # df = df.dropna(subset=["sig_b", subset_group], how="any").reset_index(drop=True)
    df = df.dropna(subset=["sig_b"], how="any").reset_index(drop=True)

    df = set_heterogeneity(df)
    df = df.loc[df["heterogeneity_category"].isin(["Homogeen", "Heterogeen", "Matig heterogeen"])].reset_index(drop=True)
    # df = df.loc[df["heterogeneity_category"] != "Matig heterogeen"]
    # df = df.loc[df["heterogeneity_category"] == "Heterogeen"]
    df = df.loc[df["heterogeneity_category"] == "Homogeen"]

    lr_results = {}
    df_years = []
    for year_range in year_ranges:
        df_year = df.loc[(df["construction_year"] >= year_range[0]) & (df["construction_year"] <= year_range[1])]
        lr_results[f"{year_range[0]}-{year_range[1]}"] = fit_linear_regression_group(df_year, subset_group=subset_group, regress_HR=regress_HR)
        df_year[subset_group] = f"{year_range[0]}-{year_range[1]}"
        df_years.append(df_year)
    df_years = pd.concat(df_years, axis=0)

    y = [val["target_training"] for val in lr_results.values()]
    y = np.array([x for sublist in y for x in sublist])

    y_hat = [val["fitted_values_training"] for val in lr_results.values()]
    y_hat = np.array([x for sublist in y_hat for x in sublist])

    # r_2 = r2_score(y, y_hat)
    # rmse = np.sqrt(mean_squared_error(y, y_hat))

    g = sns.lmplot(
        x="age_at_investigation",
        y="target",
        hue=subset_group,
        data=df_years,
        height=5,
        aspect=1.2,
        palette="Set1",
    )

    g.set_axis_labels("Leeftijd [jaar]", "${\sigma}_{b}$ [kPa]")
    # g.fig.suptitle(r"${R}^{2}$=" + f"{r_2 * 100:.0f}%" + f"\nRMSE={rmse:.2f} [kPa]")

    g._legend.set_bbox_to_anchor((.1, .1))
    g._legend.set_loc("lower left")

    plt.tight_layout()
    g.fig.subplots_adjust(top=0.85)

    g.fig.savefig(plot_path / f"Regression construction year groups.png", dpi=600)
    plt.close(g.fig)

    for (key, val) in lr_results.items():
        pval = val["p_values"][-1]
        print(f"Category {key}, p-value={pval}, statistical significance: {pval<=0.05}")

    pass


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--n_groups", type=int, default=5)
    parser.add_argument("--log_y", action="store_true")
    parser.add_argument("--regress_HR", action="store_true")
    args = parser.parse_args()

    main(
        n_groups=args.n_groups,
        log_y=args.log_y,
        # log_y=True,
        # regress_HR=args.regress_HR
        regress_HR=False
    )

    # for subset in ["HR", "bitumen"]:
    #     for n_groups in [2, 3, 5, 8, 10, 20]:
    #         print(f"Subset: {subset} and {n_groups} groups")
    #         main(
    #             n_groups=n_groups,
    #             subset=subset,
    #             bin_method=args.bin_method,
    #             log_y=False,
    #             verbose=True
    #         )
    #         print("\n")

