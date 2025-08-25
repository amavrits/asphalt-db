import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.ml.xgboost_model import *
from main.ml.xgboost.xgboost_utils import *
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import argparse


def empty_dir(dir):
    for item in dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def run(n_splits, use_bitumen, params, log_y):

    SCRIPT_PATH = Path(__file__).parent
    if args.use_bitumen:
        BASE_DATA_PATH = SCRIPT_PATH.parent.parent.parent / "data/from_bernadette/gold/w_bitumen"
        BASE_RESULT_PATH = SCRIPT_PATH.parent.parent.parent / "results/ml/xgboost/w_bitumen"
    else:
        BASE_DATA_PATH = SCRIPT_PATH.parent.parent.parent / "data/from_bernadette/gold/wo_bitumen"
        BASE_RESULT_PATH = SCRIPT_PATH.parent.parent.parent / "results/ml/xgboost/wo_bitumen"

    BASE_RESULT_PATH.mkdir(exist_ok=True, parents=True)

    BASE_RESULT_PATH = BASE_RESULT_PATH / f"logy_{str(log_y)}"

    run_splits(
        data_path=BASE_DATA_PATH,
        base_result_path=BASE_RESULT_PATH,
        n_splits=n_splits,
        params=params,
        log_y=log_y,
    )

    compile_splits(BASE_RESULT_PATH)


def run_splits(data_path, base_result_path, n_splits, params, log_y=False):
    splits = sorted([p for p in data_path.iterdir() if p.is_dir()])

    random.seed(42)
    idx_splits_to_run = random.choices(range(1, len(splits)), k=args.n_splits)
    splits_to_run = [splits[i] for i in idx_splits_to_run]

    for i_split, split in enumerate(tqdm(splits_to_run)):
        result_path = base_result_path / f"{split.stem}"
        result_path.mkdir(parents=True, exist_ok=True)
        model_predictions = run_pipeline(
            data_path=split,
            result_path=result_path,
            params=params,
            log_y=log_y,
            verbose=False
        )


def compile_splits(base_path):
    split_folders = [f for f in base_path.iterdir() if f.is_dir()]

    results = {}
    for split_folder in split_folders:
        split_key = split_folder.stem

        with open(split_folder / "model_predictions.json", "r") as f:
            model_predictions = json.load(f)

        with open(split_folder / "lr_predictions.json", "r") as f:
            lr_predictions = json.load(f)

        split_results = {
            "y_train": model_predictions["mean"]["y_train"],
            "y_test": model_predictions["mean"]["y_test"],
            "y": model_predictions["mean"]["y"],
            "model_pred_train": model_predictions["mean"]["y_pred_train"],
            "model_pred_test": model_predictions["mean"]["y_pred_test"],
            "model_pred_all": model_predictions["mean"]["y_pred_all"],
            "lr_pred_train": lr_predictions["y_pred_train"],
            "lr_pred_test": lr_predictions["y_pred_test"],
            "lr_pred_all": lr_predictions["y_pred_all"],
            "model_r2_train": model_predictions["mean"]["r2_train"],
            "model_r2_test": model_predictions["mean"]["r2_test"],
            "model_r2_all": model_predictions["mean"]["r2_all"],
            "lr_r2_train": lr_predictions["r2_train"],
            "lr_r2_test": lr_predictions["r2_test"],
            "lr_r2_all": lr_predictions["r2_all"]
        }

        results.update({split_key: split_results})

    with open(base_path / "compiled_results.json", "w") as f:
        json.dump(results, f, indent=4)

    lr_r2s = [[val["lr_r2_train"], val["lr_r2_test"], val["lr_r2_all"]] for val in results.values()]
    model_r2s = [[val["model_r2_train"], val["model_r2_test"], val["model_r2_all"]] for val in results.values()]

    lr_r2_avg = np.array(lr_r2s).mean(axis=0).tolist()
    model_r2_avg = np.array(model_r2s).mean(axis=0).tolist()
    avg_lines = [
        "-------------- Average R² per model and dataset ---------------------_---\n",
        f"{'':20} {'Training set':>15} {'Test set':>15} {'Entire dataset':>20}\n",
        f"{'LR':20} {lr_r2_avg[0]:>15.2f} {lr_r2_avg[1]:>15.2f} {lr_r2_avg[2]:>20.2f}\n",
        f"{'Model':20} {model_r2_avg[0]:>15.2f} {model_r2_avg[1]:>15.2f} {model_r2_avg[2]:>20.2f}\n"
    ]

    lr_r2_minmax = np.vstack((np.array(lr_r2s).min(axis=0), np.array(lr_r2s).max(axis=0))).tolist()
    model_r2_minmax = np.vstack((np.array(model_r2s).min(axis=0), np.array(model_r2s).max(axis=0))).tolist()
    minmax_lines = [
        "---------------------------- Min-Max R² per model and dataset -----------------------------------\n",
        f"{'':20} {'Training set':>15} {'Test set':>21} {'Entire dataset':>21}\n",
        f"{'LR':20} "
        f"{lr_r2_minmax[0][0]:>10.2f}–{lr_r2_minmax[1][0]:<10.2f} "
        f"{lr_r2_minmax[0][1]:>10.2f}–{lr_r2_minmax[1][1]:<10.2f} "
        f"{lr_r2_minmax[0][2]:>10.2f}–{lr_r2_minmax[1][2]:<10.2f}\n",
        f"{'Model':20} "
        f"{model_r2_minmax[0][0]:>10.2f}–{model_r2_minmax[1][0]:<10.2f} "
        f"{model_r2_minmax[0][1]:>10.2f}–{model_r2_minmax[1][1]:<10.2f} "
        f"{model_r2_minmax[0][2]:>10.2f}–{model_r2_minmax[1][2]:<10.2f}\n"
    ]

    lines = avg_lines + ["\n"] + minmax_lines

    for line in lines:
        print(line, end="")

    split_lines = [
        f"Split {key} - Test:  LR R²={val['lr_r2_test']:.2f}    Model R²={val['model_r2_test']:.2f}\n"
        for (key, val) in results.items()
    ]
    with open(base_path / "r2s.txt", "w") as f:
        f.writelines(lines + ["\n"] + split_lines)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits", type=int, default=2)
    parser.add_argument("--use_bitumen", action="store_true")
    parser.add_argument("--log_y", action="store_true")
    args = parser.parse_args()

    params = {
        'n_estimators': [500, 1000, 2000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1.0, 1.5]
    }

    run(
        n_splits=args.n_splits,
        use_bitumen=args.use_bitumen,
        params=params,
        log_y=args.log_y,
    )

