import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import pytensor.tensor as pt
from sklearn.metrics import r2_score
from pathlib import Path


class HBModel:

    def __init__(self, model_type: str):
        self.model_type = model_type

    def set_pooled_model(self, X, y_obs):

        with pm.Model() as model:

            intercept = pm.HalfNormal("intercept", sigma=30)
            slope = pm.HalfNormal("slope", sigma=10, shape=(2,))

            y_hat = pm.Normal("y_hat", intercept - pt.sum(slope * X, axis=1))

            sigma = pm.HalfNormal("sigma", sigma=10)

            y = pm.Normal("obs", mu=y_hat, sigma=sigma, observed=y_obs)

        return model

    def set_unpooled_model(self, X, y_obs, group_idx):

        n_groups = len(np.unique(group_idx))

        with pm.Model() as model:

            intercept = pm.HalfNormal("intercept", sigma=30, shape=(n_groups,))
            slope = pm.Normal("slope", sigma=10, shape=(n_groups, 2))

            y_hat = pm.Normal("y_hat", intercept[group_idx] - pt.sum(slope[group_idx] * X, axis=1))

            sigma = pm.HalfNormal("sigma", sigma=10)

            y = pm.Normal("obs", mu=y_hat, sigma=sigma, observed=y_obs)

        return model

    def set_partially_pooled_model(self, X, y_obs, group_idx):

        n_groups = len(np.unique(group_idx))

        with pm.Model() as model:

            intercept_mean = pm.HalfNormal("intercept_mean", sigma=20)
            intercept_sd = pm.HalfNormal("intercept_sd", sigma=4)

            slope_mean = pm.HalfNormal("slope_mean", sigma=10)
            slope_sd = pm.HalfNormal("slope_sd", sigma=2)

            intercept_offset = pm.TruncatedNormal("intercept_offset", mu=0, sigma=1, lower=-intercept_mean/intercept_sd, shape=(n_groups,))
            intercept = pm.Deterministic("intercept", intercept_mean + intercept_sd * intercept_offset)

            slope_offset = pm.TruncatedNormal("slope_offset", mu=0, sigma=1, lower=-slope_mean / slope_sd, shape=(n_groups, 2))
            slope = pm.Deterministic("slope", slope_mean + slope_sd * slope_offset)

            y_hat = pm.Normal("y_hat", intercept[group_idx] + pt.sum(slope[group_idx] * X, axis=1))

            sigma = pm.HalfNormal("sigma", sigma=10)

            y = pm.Normal("obs", mu=y_hat, sigma=sigma, observed=y_obs)

        return model

    def train(self, X, y, group_idx=None, path=None):

        self.X = X
        self.y = y
        self.group_idx = group_idx

        if self.model_type == "pooled":
            model = self.set_pooled_model(X, y)
        elif self.model_type == "unpooled":
            model = self.set_unpooled_model(X, y, group_idx)
        elif self.model_type == "partially_pooled":
            model = self.set_partially_pooled_model(X, y, group_idx)

        if path is not None:
            if not Path(path/f"{self.model_type}_idata.netcdf").is_file():
                self.idata = pm.sample(model=model, draws=1_000, tune=1_000, progressbar=True)
                self.idata.to_netcdf(path / f"{self.model_type}_idata.netcdf")
            else:
                self.idata = az.from_netcdf(path/f"{self.model_type}_idata.netcdf")
        else:
           self.idata = pm.sample(model=model, draws=1_000, tune=1_000, progressbar=True)

    def predict(self, X):

        intercept = self.idata["posterior"].intercept.values
        if self.model_type == "pooled":
            intercept = intercept[..., np.newaxis]
        n_groups = intercept.shape[-1]
        intercept = intercept.reshape(-1, n_groups)
        slope = self.idata["posterior"].slope.values.reshape(-1, n_groups, X.shape[-1])

        y_hat = intercept - np.sum(slope[np.newaxis, ...] * X[:, np.newaxis, np.newaxis], axis=-1)

        y_hat = y_hat.transpose(1, 2, 0)
        y_hat_mean = y_hat.mean(axis=0)
        y_hat_quantiles = np.quantile(y_hat, [0.05, 0.95], axis=0)

        return y_hat_mean, y_hat_quantiles

    def eval(self):

        y_hat, _ = self.predict(self.X)
        if self.model_type != "pooled":
            y_hat = y_hat[self.group_idx, np.arange(y_hat.shape[-1])]

        r2 = r2_score(self.y, y_hat)

        self.r2 = r2


if __name__ == "__main__":

    pass

