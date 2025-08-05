import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from src.ml.utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.0):
        """
        input_dim: int, number of input features.
        hidden_layers: list of ints, sizes of hidden layers.
        dropout_rate: float, dropout rate applied after each hidden layer.
        """
        super(MLPRegressor, self).__init__()
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def fit(self, X, y, epochs=100, lr=1e-4):

        self.x_scaler = MinMaxScaler()
        X_scaled = self.x_scaler.fit_transform(X)

        self.y_scaler = MinMaxScaler()
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).squeeze()

        # Convert to PyTorch Tensors
        X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_scaled_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)

        # Loss & Optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in tqdm(range(epochs)):
            self.train()
            optimizer.zero_grad()
            outputs = self.forward(X_scaled_tensor)
            loss = criterion(outputs, y_scaled_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    def predict(self, X):
        self.eval()
        if isinstance(X, np.ndarray):
            X = torch.tensor(X)
        X_scaled = self.x_scaler.transform(X)
        X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            y_scaled = self.forward(X_scaled_tensor)
        y = self.y_scaler.inverse_transform(y_scaled.cpu().numpy())
        return y


def predict(model, X_train, y_train, X_test, y_test):

    X = np.vstack((X_train, X_test))
    y = np.concat((y_train, y_test))

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_all = model.predict(X)

    idx = np.argsort(y)[::-1]
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

