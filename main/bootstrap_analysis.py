import pandas as pd
import numpy as np
from scipy.stats import t
from pathlib import Path
import json
from dataclasses import dataclass, field, asdict
from numpy.typing import NDArray
from typing import List, Dict, Tuple, Optional
from argparse import ArgumentParser
import matplotlib.pyplot as plt


@dataclass
class BootstrapResults:
    x: NDArray
    y: NDArray
    ages: NDArray
    n_bootstrap: int = 10_000
    ci_lvl: float = 0.95
    two_tailed: bool = True
    boot_mean: NDArray= field(init=False)
    boot_quantile: NDArray= field(init=False)
    boot_slope: NDArray= field(init=False)
    age_diff: int | float = field(init=False)
    mean_ci: NDArray = field(init=False)
    quantile_ci: NDArray = field(init=False)
    slope_ci: NDArray = field(init=False)

    def __post_init__(self) -> None:
        alpha = (1 - self.ci_lvl) / 2
        self.age_diff = np.abs(np.diff(self.ages)).item()
        self.bootstrap()
        self.mean_ci = np.quantile(self.boot_mean, q=[alpha, 1-alpha])
        self.quantile_ci = np.quantile(self.boot_quantile, q=[alpha, 1-alpha])
        self.slope_ci = np.quantile(self.boot_slope, q=[alpha, 1-alpha])

    def bootstrap(self):

        n_samples = self.y.size
        idxs = np.arange(self.x.size)
        idx_samples = np.random.choice(idxs, replace=True, size=(self.n_bootstrap, n_samples))

        x_boot = self.x[idx_samples]

        self.boot_mean = x_boot.mean(axis=1)

        boot_std = np.std(x_boot, ddof=1, axis=1)
        self.boot_quantile = t(loc=self.boot_mean, scale=boot_std, df=n_samples-1).ppf(0.05)

        boot_mean_diff = self.y.mean() - self.boot_mean
        self.boot_slope = boot_mean_diff / self.age_diff

    def to_dict_of_lists(self) -> None:
        d = asdict(self)
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def select_dike(df: pd.DataFrame, dike_id: str = "emmapolder") -> str:
    available_dikes = pd.unique(df["dike_name"]).tolist()
    dike_name = [d for d in available_dikes if dike_id in d]
    if len(dike_name) > 1:
        raise ValueError(f"Invalid dike ID, '{dike_id}' is present in multiple dike names {available_dikes}.")
    else:
        dike_name = dike_name[0]
    return dike_name


def filter_data(df: pd.DataFrame, dike_name: str) -> pd.DataFrame:
    df_filtered = df.copy()
    df_filtered = df_filtered.loc[df_filtered["dike_name"] == dike_name]
    df_filtered = df_filtered.reset_index(drop=True)
    df_filtered["age"] = df_filtered["investigation_year"] - df_filtered["construction_year"]
    columns_keep = ["age", "sig_b"]
    df_filtered = df_filtered[columns_keep]
    return df_filtered


def bootstrap(
        x: NDArray,
        y: NDArray,
        ages: NDArray,
        n_bootstrap: int = 1_000,
        ci_lvl: float = .95
) -> BootstrapResults:

    n_samples = y.size
    idxs = np.arange(x.size)
    idx_samples = np.random.choice(idxs, replace=True, size=(n_bootstrap, n_samples))

    x_boot = x[idx_samples]

    boot_mean = x_boot.mean(axis=1)
    boot_mean_diff = y.mean() - boot_mean

    age_diff = np.abs(np.diff(ages))
    boot_slope = boot_mean_diff / age_diff

    return BootstrapResults(x=x, y=y, ages=ages, ci_lvl=ci_lvl)


def bootstrap_analysis(df: pd.DataFrame, n_bootstrap: int = 1_000, ci_lvl: float = .95) -> Dict[int, Dict[str, float]]:

    ages = df["age"].values
    str = df["sig_b"].values

    age_counts = np.unique_counts(ages)

    n_min_count_age = age_counts.counts.min()
    min_count_age = age_counts.values[age_counts.counts == n_min_count_age].item()
    str_min_count_age = str[ages == min_count_age]

    def mean_stat(x, axis=0):
        np.mean(x, axis=axis)

    def mean_diff_stat(x, y, axis=0):
        np.mean(x, axis=axis) - np.mean(y, axis=axis)

    age_bootstrap_res = {}
    for age in age_counts.values:

        if age == min_count_age:
            continue

        age_diff = age - min_count_age # Assumes minimum number of observations at earliest age
        str_current_age = str[ages==age]

        bootstrap_res = BootstrapResults(
            x=str_current_age,
            y=str_min_count_age,
            ages=np.array([min_count_age, age]),
            n_bootstrap=n_bootstrap,
            ci_lvl=ci_lvl
        )

        age_bootstrap_res[int(age)] = bootstrap_res

    return age_bootstrap_res


def plot_bootstrap(bootstrap_res: BootstrapResults, path: Path) -> None:

    fig = plt.figure()

    plt.scatter([bootstrap_res.ages.min()], [bootstrap_res.y.mean()], marker="x", c="b", label="Observation mean")
    plt.scatter([bootstrap_res.ages.max()], [bootstrap_res.x.mean()], marker="x", c="r", label="Observation mean")

    mean = bootstrap_res.boot_mean.mean()
    errorbar = np.array([mean-bootstrap_res.mean_ci.min(), bootstrap_res.mean_ci.max()-mean])[:, np.newaxis]
    plt.errorbar(x=bootstrap_res.ages.max(), y=mean, yerr=errorbar, c="r", label="Mean bootstrapped\n95% CI", capsize=5, fmt='o')

    mean = bootstrap_res.boot_quantile.mean()
    errorbar = np.array([mean-bootstrap_res.quantile_ci.min(), bootstrap_res.quantile_ci.max()-mean])[:, np.newaxis]
    plt.errorbar(x=bootstrap_res.ages.max(), y=mean, yerr=errorbar, c="orange", label="0.05 quantile\nbootstrapped 95% CI", capsize=5, fmt='o')

    slope_quantiles = np.quantile(bootstrap_res.boot_slope, q=[0.025, 0.975])
    str_quantiles = bootstrap_res.y.mean() - slope_quantiles * bootstrap_res.ages
    str_quantiles = np.vstack((np.array([bootstrap_res.y.mean(), bootstrap_res.y.mean()]), str_quantiles)).T
    plt.fill_between(x=bootstrap_res.ages, y1=str_quantiles[0], y2=str_quantiles[1], color="r", alpha=0.4)

    plt.xlabel("Age", fontsize=14)
    plt.ylabel("Flectural strength [kPa]", fontsize=14)
    plt.grid()
    plt.legend(fontsize=10)
    plt.close()
    fig.savefig(path/"age_plot.png")


    fig, axs = plt.subplots(1, 3, figsize=(16, 6))

    ax = axs[0]
    ax.hist(bootstrap_res.boot_mean, density=True, color="b", alpha=0.4, bins=100)
    errorbar = np.array([
        bootstrap_res.boot_mean.mean()-bootstrap_res.mean_ci.min(),
        bootstrap_res.mean_ci.max()-bootstrap_res.boot_mean.mean()]
    )[:, np.newaxis]
    height = 0.6 * ax.get_ylim()[1]
    ax.errorbar(x=bootstrap_res.boot_mean.mean(), xerr=errorbar, y=height, c="b", label="95% CI", capsize=5, fmt='o')
    ax.set_xlabel("Mean [kPa]", fontsize=14)
    ax.set_ylabel("Density [-]", fontsize=14)
    ax.set_title("Boostrap of mean")

    ax = axs[1]
    ax.hist(bootstrap_res.boot_quantile, density=True, color="b", alpha=0.4, bins=100)
    errorbar = np.array([
        bootstrap_res.boot_quantile.mean()-bootstrap_res.quantile_ci.min(),
        bootstrap_res.quantile_ci.max()-bootstrap_res.boot_quantile.mean()]
    )[:, np.newaxis]
    height = 0.6 * ax.get_ylim()[1]
    ax.errorbar(x=bootstrap_res.boot_quantile.mean(), xerr=errorbar, y=height, c="b", label="95% CI", capsize=5, fmt='o')
    ax.set_xlabel("0.05 Quantile [kPa]", fontsize=14)
    ax.set_ylabel("Density [-]", fontsize=14)
    ax.set_title("Boostrap of 0.05 quantile")

    ax = axs[2]
    ax.hist(bootstrap_res.boot_slope, density=True, color="b", alpha=0.4, bins=100)
    errorbar = np.array([
        bootstrap_res.boot_slope.mean()-bootstrap_res.slope_ci.min(),
        bootstrap_res.slope_ci.max()-bootstrap_res.boot_slope.mean()]
    )[:, np.newaxis]
    height = 0.6 * ax.get_ylim()[1]
    ax.errorbar(x=bootstrap_res.boot_slope.mean(), xerr=errorbar, y=height, c="b", label="95% CI", capsize=5, fmt='o')
    ax.set_xlabel("0.05 Quantile [kPa]", fontsize=14)
    ax.set_ylabel("Density [-]", fontsize=14)
    ax.set_title("Boostrap of 0.05 quantile")

    plt.close()
    fig.savefig(path/"histograms.png")


def main(dike_id: str = "emmapolder", n_bootstrap: int = 10_000) -> None:

    script_path = Path(__file__).parent
    data_path = script_path.parent / "data/db_querry.csv"

    df = pd.read_csv(data_path)

    df_multiple_projects = df.groupby("dike_name").filter(lambda x: x["project_name"].nunique() >= 2)
    df_multiple_projects = df_multiple_projects.sort_values(by=["dike_name", "project_name"])
    df_multiple_projects = df_multiple_projects.reset_index(drop=True)

    dike_name = select_dike(df=df_multiple_projects, dike_id=dike_id)

    result_path = script_path.parent / f"results/bootstrapping/{dike_name}"
    result_path.mkdir(parents=True, exist_ok=True)

    df_filtered = filter_data(df=df_multiple_projects, dike_name=dike_name)

    age_bootstrap_res = bootstrap_analysis(df=df_filtered, n_bootstrap=n_bootstrap)

    for (age, res) in age_bootstrap_res.items():
        plot_path = result_path / f"age_{age}/plots"
        plot_path.mkdir(parents=True, exist_ok=True)
        plot_bootstrap(res, plot_path)
        age_bootstrap_res[age] = res.to_dict_of_lists()

    with open(result_path/"bootstrap_results.json", "w") as f:
        json.dump(age_bootstrap_res, f, indent=4)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dike_id", type=str, default="emmapolder")
    parser.add_argument("--n_bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    main(
        dike_id=args.dike_id,
        n_bootstrap=args.n_bootstrap
    )

