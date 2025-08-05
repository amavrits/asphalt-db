import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from src.ml.utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm


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


if __name__ == "__main__":

    # data_path = os.environ["DATA_PATH"]
    SCRIPT_DIR = Path(__file__).parent
    data_path = SCRIPT_DIR.parent.parent / "data"
    data_file = data_path / "from_bernadette/Database WAB - overzicht ADL28062023_BWich_selection18.8.xlsx"
    result_path = Path("../results")
    result_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(data_file)

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    x_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).squeeze()

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Initialize Model
    hidden_layers = [256, 128, 64, 32]
    # hidden_layers = [64, 32, 16]
    model = MLPRegressor(input_dim=X.shape[1], hidden_layers=hidden_layers, dropout_rate=0.2)

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 30_000
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Predictions
    model.eval()
    with torch.no_grad():
        y_hat_scaled = model(X_train_tensor).numpy()
        y_pred_scaled = model(X_test_tensor).numpy()

    # Inverse transform to original scale
    y_hat = y_scaler.inverse_transform(y_hat_scaled).squeeze()
    y_pred = y_scaler.inverse_transform(y_pred_scaled).squeeze()

    fig = plt.figure()
    plt.scatter(y_train, y_hat, c="b", label="Train")
    plt.scatter(y_test, y_pred, c="r", label="Test")
    plt.axline([0, 0], slope=1, c="k", linestyle="--")
    plt.xlabel("Observations", fontsize=12)
    plt.ylabel("Predictions", fontsize=12)
    plt.legend()
    plt.suptitle(
        "Training ${R}^{2}$="+f"{r2_score(y_train, y_hat):.2f}\n"+
        "Testing ${R}^{2}$=" + f"{r2_score(y_test, y_pred):.2f}"
    )
    plt.show()

