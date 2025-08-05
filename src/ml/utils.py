import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def prepare_data(df):
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

    # Multiplied features
    df['age_x_void'] = df['age'] * df['void_ratio']
    df['age_x_bitumen'] = df['age'] * df['bitumen']
    df['void_x_bitumen'] = df['void_ratio'] * df['bitumen']
    df['bitumen_per_void'] = df['bitumen'] / (df['void_ratio'] + 1e-6)  # Avoid division by zero

    # Polynomial Features
    df['age_squared'] = df['age'] ** 2
    df['void_squared'] = df['void_ratio'] ** 2
    df['bitumen_squared'] = df['bitumen'] ** 2

    # Log Features
    df['log_age'] = np.log1p(df['age'])
    df['log_void'] = np.log1p(df['void_ratio'])
    df['log_bitumen'] = np.log1p(df['bitumen'])

    # Reciprocal Features
    df['inv_age'] = 1 / (df['age'] + 1)
    df['inv_void'] = 1 / (df['void_ratio'] + 1)

    # Mean of all features (as aggregate)
    df['mean_feature'] = df[['age', 'void_ratio', 'bitumen']].mean(axis=1)

    df = df.dropna(how="any")

    y = df["str"].values
    df = df.drop(columns=["dijk", "project", "str"])
    X = df.values

    return X, y


def get_data(df):
    y = df["str"].values
    df = df.drop(columns=["dijk", "project", "str"])
    X = df.values
    return X, y


def lr_model(X_train, y_train, X_test, y_test):

    X = np.vstack((X_train, X_test))
    y = np.concat((y_train, y_test))

    def f(x):
        age = x[:, 0]
        void_ratio = x[:, 1]
        y = np.where(
            age <= 40,
            10.5852 - 0.0054 * age ** 2 + 8.341e-5 * age ** 3 - 0.3077 * void_ratio,
            6.8238 - 0.0466 * void_ratio ** 2 + 0.0026 * void_ratio ** 3 - 5.17e-6 * void_ratio ** 2 * age ** 2
        )
        return y

    y_pred_train = f(X_train)
    y_pred_test = f(X_test)
    y_pred_all = f(X)

    idx = np.argsort(y)
    y = y[idx]
    y_pred_all = y_pred_all[idx]

    test_flag = np.zeros(y.size).astype(bool)
    test_flag[-y_test.size:] = 1
    test_flag = test_flag[idx]

    predictions = {
        "y_train": y_train,
        "y_test": y_test,
        "y": y,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_pred_all": y_pred_all,
        "test_flag": test_flag,
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "r2_all": r2_score(y, y_pred_all),
    }

    return predictions


def plot_predictions(predictions, lr_predictions, plot_path, model_name):

    fig, axs = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)

    y_train = lr_predictions["y_train"]
    y_test = lr_predictions["y_test"]
    y_pred_train = lr_predictions["y_pred_train"]
    y_pred_test = lr_predictions["y_pred_test"]

    axs[0].scatter(y_train, y_pred_train, c="b", label="Train")
    axs[0].scatter(y_test, y_pred_test, c="r", label="Test")
    axs[0].axline([0, 0], slope=1, c="k", linestyle="--")
    axs[0].set_xlabel("Observations", fontsize=12)
    axs[0].set_ylabel("Predictions", fontsize=12)
    axs[0].legend()
    axs[0].set_title(
        "Linear Regression Model\n" +
        f"Training $R^2$ = {lr_predictions['r2_train']:.2f}\n" +
        f"Testing $R^2$ = {lr_predictions['r2_test']:.2f}\n" +
        f"Entire dataset $R^2$ = {lr_predictions['r2_all']:.2f}"
    )

    y_train = predictions["y_train"]
    y_test = predictions["y_test"]
    y_pred_train = predictions["y_pred_train"]
    y_pred_test = predictions["y_pred_test"]

    axs[1].scatter(y_train, y_pred_train, c="b", label="Train")
    axs[1].scatter(y_test, y_pred_test, c="r", label="Test")
    axs[1].axline([0, 0], slope=1, c="k", linestyle="--")
    axs[1].set_xlabel("Observations", fontsize=12)
    axs[1].set_ylabel("Predictions", fontsize=12)
    axs[1].legend()
    axs[1].set_title(
        f"{model_name}\n" +
        f"Training $R^2$ = {predictions['r2_train']:.2f}\n" +
        f"Testing $R^2$ = {predictions['r2_test']:.2f}\n" +
        f"Entire dataset $R^2$ = {predictions['r2_all']:.2f}"
    )

    fig.savefig(plot_path/"predictions.png")


def plot_quantiles(predictions, plot_path):

    mean_predictions = predictions["mean"]
    lower_predictions = predictions["lower"]
    upper_predictions = predictions["upper"]

    y = mean_predictions["y"]
    y_pred_all = mean_predictions["y_pred_all"]
    test_flag = mean_predictions["test_flag"]

    idx = np.argsort(y)
    y = y[idx]
    y_pred_all = y_pred_all[idx]
    test_flag = test_flag[idx]

    y_err_lower = np.abs(y_pred_all - lower_predictions["y_pred_all"][idx])
    y_err_upper = np.abs(y_pred_all - upper_predictions["y_pred_all"][idx])

    x = np.arange(1, y.size + 1)

    points_per_plot = 40
    n_plots = y.size // points_per_plot + 1

    for i in range(n_plots):

        plot_idx = np.arange(points_per_plot*i, min(points_per_plot*(i+1), y.size-1))

        fig = plt.figure(figsize=(16, 6))
        plt.scatter(x[plot_idx][~test_flag[plot_idx]], mean_predictions["y"][plot_idx][~test_flag[plot_idx]], color="b", marker="x", label="Training data")
        plt.errorbar(
            x[plot_idx][~test_flag[plot_idx]],
            y_pred_all[plot_idx][~test_flag[plot_idx]],
            yerr=[y_err_lower[plot_idx][~test_flag[plot_idx]], y_err_upper[plot_idx][~test_flag[plot_idx]]],
            fmt="o", ecolor="b", alpha=0.7, capsize=3, label="Training 90% PI")
        plt.scatter(x[plot_idx][test_flag[plot_idx]], mean_predictions["y"][plot_idx][test_flag[plot_idx]], color="r", marker="x", label="Testing data")
        plt.errorbar(
            x[plot_idx][test_flag[plot_idx]],
            y_pred_all[plot_idx][test_flag[plot_idx]],
            yerr=[y_err_lower[plot_idx][test_flag[plot_idx]], y_err_upper[plot_idx][test_flag[plot_idx]]],
            fmt="o", ecolor="r", alpha=0.7, capsize=3, label="Testing 90% PI")
        plt.xlabel("Point ID", fontsize=12)
        plt.ylabel("Strength [kPa]", fontsize=12)
        plt.grid()
        fig.savefig(plot_path/f"quantiles_{min(plot_idx)+1}-{max(plot_idx)+1}.png")


if __name__ == "__main__":

    pass

