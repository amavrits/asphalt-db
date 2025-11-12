import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import json
from argparse import ArgumentParser
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style("whitegrid")



def fit_linear_regression_group(df: pd.DataFrame, subset_group: str = "HR_group", group: str = "all") -> Dict[str, Any]:

    df_training = df.loc[df[subset_group] == group].reset_index(drop=True)

    X = df_training["age_at_investigation"]
    X = sm.add_constant(X)
    y = df_training["target"]

    model = sm.OLS(y, X).fit()
    summary = model.summary()

    prediction_training = model.get_prediction(sm.add_constant(df_training["age_at_investigation"]))
    summary_prediction_training = prediction_training.summary_frame(alpha=0.05)
    mean_prediction_training = summary_prediction_training["mean"].values
    ci_prediction_training = summary_prediction_training[["mean_ci_lower", "mean_ci_upper"]].values
    pi_prediction_training = summary_prediction_training[["obs_ci_lower", "obs_ci_upper"]].values

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
        "X_training": df_training["age_at_investigation"].values,
        "X_all": df["age_at_investigation"].values,
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
        subset: str = "HR",
        log_y: bool = True
) -> None:

    script_path = Path(__file__).parent
    data_path = script_path.parent / "data/database_all_v3.csv"
    # data_path = script_path.parent / "data/database_all_v5.csv"
    result_path = script_path.parent / f"results/categorized_regression/log_y_{log_y}"
    result_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    df = df.dropna(subset=["sig_b"]).reset_index(drop=True)
    if log_y:
        df["target"] = np.log(df["sig_b"])
    else:
        df["target"] = df["sig_b"]

    if subset == "bitumen":
        df = df.dropna(subset=[subset]).reset_index(drop=True)

    subset_category = f"{subset}_category"
    subset_group = f"{subset}_group"

    bins = np.linspace(df[subset].min(), df[subset].max(), n_groups+1).tolist()
    labels = [f"{i:.1f}-{j:.1f}" for (i, j) in zip(bins[:-1], bins[1:])]
    df[subset_category] = pd.cut(df[subset], bins=bins, labels=labels, right=False)
    df[subset_group] = df[subset_category].cat.codes

    lr_results = {}
    for group in sorted(pd.unique(df[subset_group]).tolist()):
        if len(pd.unique(df.loc[df[subset_group] == group, "age_at_investigation"])) <= 1:
            continue
        lr_results[labels[group]] = fit_linear_regression_group(df, subset_group=subset_group, group=group)

    y = [val["target_training"] for val in lr_results.values()]
    y = np.array([x for sublist in y for x in sublist])

    y_hat = [val["fitted_values_training"] for val in lr_results.values()]
    y_hat = np.array([x for sublist in y_hat for x in sublist])

    r_2 = r2_score(y, y_hat)
    rmse = np.sqrt(mean_squared_error(y, y_hat))


    fig = plt.figure()
    sns.lmplot(
        x="age_at_investigation",
        y="target",
        hue=subset_category,
        data=df,
        height=5,
        aspect=1.2,
        palette="Set1"
    )
    plt.xlabel("Age [yr]")
    plt.ylabel("${σ}_{b}$ [kPa]")
    plt.suptitle("${R}^{2}$="+f"{r_2*100:.0f}%\nRMSE={rmse:.2f} [kPa]")
    plt.show()

    pass


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--n_groups", type=int, default=5)
    parser.add_argument("--subset", type=str, default="HR")
    parser.add_argument("--log_y", action="store_false")
    args = parser.parse_args()

    main(
        n_groups=args.n_groups,
        subset=args.subset,
        # subset="bitumen",
        # log_y=args.log_y
        log_y=False
    )

