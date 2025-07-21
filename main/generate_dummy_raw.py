import pandas as pd
import numpy as np
from pathlib import Path
import json


if __name__ == "__main__":

    n = 100
    n_bhs = 5
    n_samples = 10

    SCRIPT_DIR = Path(__file__).parent

    for data_type in ["raw", "processed", "summarized"]:

        with open(SCRIPT_DIR.parent/f"data/{data_type}_columns.json", "r") as f:
            test = json.load(f)

        for (test_name, test_columns) in test.items():

            for bh in range(1, n_bhs+1):

                data_path = SCRIPT_DIR.parent / f"data/dummy/{data_type}_data/{test_name}/{bh}"
                data_path.mkdir(exist_ok=True, parents=True)

                for sample in range(1, n_samples+1):
                    test_data = np.random.normal(size=(n, len(test_columns)))
                    df = pd.DataFrame(test_data, columns=test_columns)
                    df.to_csv(data_path/f"{sample}.csv")


