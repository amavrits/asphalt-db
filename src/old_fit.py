import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def fit_old_formula(df):
    hr = df["HR"].values
    age = df["age_at_investigation"].values
    X = np.c_[hr, age]
    y = df["sig_b"].values

    feats_young = np.c_[X[:, 1]**2, X[:, 1]**3, X[:, 0]]
    feats_young = sm.add_constant(feats_young)
    model_young = sm.OLS(y, feats_young).fit()
    feats_pred_young = np.c_[X[:, 1]**2, X[:, 1]**3, np.ones_like(hr)*4]
    feats_pred_young = sm.add_constant(feats_pred_young, has_constant='add')
    prediction = model_young.get_prediction(feats_pred_young)
    prediction_summary = prediction.summary_frame(alpha=0.10)
    mean_prediction_young = prediction_summary["mean"].values
    ci_prediction_young = prediction_summary[["mean_ci_lower", "mean_ci_upper"]].values
    pi_prediction_young = prediction_summary[["obs_ci_lower", "obs_ci_upper"]].values

    feats_old = np.c_[X[:, 0]**2, X[:, 0]**3, X[:, 0]**2*X[:, 1]**2]
    feats_old = sm.add_constant(feats_old)
    model_old = sm.OLS(y, feats_old).fit()
    feats_pred_old = np.c_[np.ones_like(hr)*4**2, np.ones_like(hr)*4**3, np.ones_like(hr)*4**2*X[:, 1]**2]
    feats_pred_old = sm.add_constant(feats_pred_old, has_constant='add')
    prediction = model_old.get_prediction(feats_pred_old)
    prediction_summary = prediction.summary_frame(alpha=0.10)
    mean_prediction_old = prediction_summary["mean"].values
    ci_prediction_old = prediction_summary[["mean_ci_lower", "mean_ci_upper"]].values
    pi_prediction_old = prediction_summary[["obs_ci_lower", "obs_ci_upper"]].values

    mean_prediction = np.where(age <= 40, mean_prediction_young, mean_prediction_old)
    ci_prediction = np.where(age[:, np.newaxis] <= 40, ci_prediction_young, ci_prediction_old)
    pi_prediction = np.where(age[:, np.newaxis] <= 40, pi_prediction_young, pi_prediction_old)

    return mean_prediction, pi_prediction


def old_formula(X):  # -----------------------------Current
    """

    :param X: 2 columns array. First column is void ratio, second columm is age.
    :return: the predicted strength
    """
    y = np.where(X[:, 1] <= 40, 10.5852 - 0.0054 * X[:, 1] ** 2 + 8.341e-05 * X[:, 1] ** 3 - 0.3077 * X[:, 0],
                 6.8238 - 0.0466 * X[:, 0] ** 2 + 0.0026 * X[:, 0] ** 3 - 5.17 * 1e-6 * X[:, 0] ** 2 * X[:, 1] ** 2)

    return y