import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from src.ml.utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt


class ProbabilisticMLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.2, device=None):
        """
        input_dim: int, number of input features.
        hidden_layers: list of ints, sizes of hidden layers.
        dropout_rate: float, dropout rate applied after each hidden layer.
        """
        super(ProbabilisticMLPRegressor, self).__init__()

        self.set_device(device)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        self.mean_head = nn.Linear(prev_dim, 1)
        self.log_std_head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        h = self.shared_layers(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        std = torch.exp(log_std)
        return mean, std

    def set_device(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        print(f"Using device: {device}")

    def nll_loss(self, y, y_hat, std):
        # Negative Log Likelihood of Normal Distribution
        var = std ** 2 + 1e-5
        return torch.mean(0.5 * torch.log(2 * np.pi * var) + 0.5 * ((y - y_hat) ** 2) / var)

    def fit(self, X, y, epochs=100, lr=1e-4):

        self.x_scaler = MinMaxScaler()
        X_scaled = self.x_scaler.fit_transform(X)

        # Convert to PyTorch Tensors
        X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        # Loss & Optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

        self.losses = []
        for epoch in tqdm(range(epochs)):
            self.train()
            optimizer.zero_grad()
            y_hat, std = self.forward(X_scaled_tensor)
            loss = self.nll_loss(y_tensor, y_hat, std)
            loss.backward()
            optimizer.step()

            self.losses.append(loss.item())
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    def predict(self, X, alpha=0.5):
        self.eval()
        X_scaled = self.x_scaler.transform(X)
        X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_hat, std = self.forward(X_scaled_tensor)
        y_hat = y_hat.cpu().numpy().squeeze()
        std = std.cpu().numpy().squeeze()
        return norm(loc=y_hat, scale=std).ppf(alpha)


def predict(model, X_train, y_train, X_test, y_test, alpha=0.5):

    X = np.vstack((X_train, X_test))
    y = np.concat((y_train, y_test))

    y_pred_train = model.predict(X_train, alpha)
    y_pred_test = model.predict(X_test, alpha)
    y_pred_all = model.predict(X, alpha)

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



if __name__ == "__main__":

    pass

