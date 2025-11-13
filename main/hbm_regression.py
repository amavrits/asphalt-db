import pandas as pd
import numpy as np
import pymc as pm
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns


def hbm_regression(df, subset_group):

    X = df["age_at_investigation"].values
    y = df["target"].values
    groups = df[subset_group].values
    n_groups = np.unique(groups).size

    with pm.Model() as hier_lm:

        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=5.0)
        mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=5.0)

        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=5.0)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=5.0)

        alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1, shape=n_groups)
        beta_offset = pm.Normal("beta_offset", mu=0, sigma=1, shape=n_groups)

        alpha_group = pm.Deterministic("alpha_group", mu_alpha+sigma_alpha*alpha_offset)
        beta_group = pm.Deterministic("beta_group", mu_beta+sigma_beta*beta_offset)

        sigma = pm.HalfNormal("sigma", sigma=5.0)

        mu = alpha_group[groups] + beta_group[groups] * X

        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=1_000,
            tune=1_000,
            target_accept=0.95,
            chains=4,
        )

    ppc = pm.sample_posterior_predictive(trace=idata, model=hier_lm, var_names=["y_obs"])
    y_pp = ppc.posterior_predictive.y_obs.values
    y_mean = y_pp.mean(axis=(0, 1))
    y_pi_lower = np.quantile(y_pp, axis=(0, 1), q=0.05)
    y_pi_upper = np.quantile(y_pp, axis=(0, 1), q=0.95)

    df["pred_mean"] = y_mean
    df["pred_lower"] = y_pi_lower
    df["pred_upper"] = y_pi_upper

    return df.copy()


def main(
        n_groups: int  = 5,
        subset: str = "HR",
        bin_method: str = "linear",
        log_y: bool = True,
        verbose: bool = False
) -> None:

    script_path = Path(__file__).parent
    data_path = script_path.parent / "data/database_all_v3.csv"
    # data_path = script_path.parent / "data/database_all_v5.csv"
    result_path = script_path.parent / f"results/HBM_regression/log_y_{log_y}/{subset}"
    result_path.mkdir(parents=True, exist_ok=True)
    plot_path = result_path / f"plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if subset == "HR_sq":
        df["HR_sq"] = df["HR"] **2

    df = df.dropna(subset=["sig_b", subset], how="any").reset_index(drop=True)

    if log_y:
        df["target"] = np.log(df["sig_b"])
    else:
        df["target"] = df["sig_b"]

    subset_category = f"{subset}_category"
    subset_group = f"{subset}_group"

    if bin_method == "linear":
        bins = np.linspace(df[subset].min(), df[subset].max(), n_groups+1).tolist()
    elif bin_method == "loglinear":
        bins = np.logspace(np.log10(df[subset].min()), np.log10(df[subset].max()), n_groups+1).tolist()
    elif bin_method == "quantiles":
        bins = np.quantile(df[subset], q=np.linspace(0, 1, n_groups+1)).tolist()

    labels = [f"{i:.1f}-{j:.1f}" for (i, j) in zip(bins[:-1], bins[1:])]
    df[subset_category] = pd.cut(df[subset], bins=bins, labels=labels, right=False)
    df[subset_group] = df[subset_category].cat.codes

    df = hbm_regression(df, subset_group)

    r_2 = r2_score(df["target"], df["pred_mean"])
    rmse = np.sqrt(mean_squared_error(df["target"], df["pred_mean"]))


    fig = plt.figure()
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in np.linspace(0, 1, n_groups)]
    for group in np.unique(df[subset_group]):
        x = df.loc[df[subset_group] == group, "age_at_investigation"].values
        idx = np.argsort(x)
        x = x[idx]
        y = df.loc[df[subset_group] == group, "target"].values[idx]
        y_hat = df.loc[df[subset_group] == group, "pred_mean"].values[idx]
        y_pi = df.loc[df[subset_group] == group, ["pred_lower", "pred_upper"]].values[idx].T
        plt.scatter(x, y, c=colors[group])
        plt.plot(x, y_hat, c=colors[group])
        plt.fill_between(x, y_pi[0], y_pi[1], color=colors[group], alpha=0.3)
    plt.xlabel("Age [yr]")
    plt.ylabel("${\sigma}_{b}$ [kPa]")
    fig.suptitle(r"${R}^{2}$=" + f"{r_2 * 100:.0f}%" + f"\nRMSE={rmse:.2f} [kPa]")
    fig.savefig(plot_path / f"Regression with {n_groups} {bin_method} groups.png", dpi=600)
    plt.close()

    pass


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--n_groups", type=int, default=5)
    parser.add_argument("--subset", type=str, default="bitumen")
    parser.add_argument("--bin_method", type=str, default="linear")
    parser.add_argument("--log_y", action="store_false")
    args = parser.parse_args()

    main(
        n_groups=args.n_groups,
        subset=args.subset,
        # subset="bitumen",
        bin_method=args.bin_method,
        # bin_method="loglinear",
        # log_y=args.log_y
        log_y=False
    )
