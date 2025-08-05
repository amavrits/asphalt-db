
source .venv/bin/activate

echo Training XGBoost
python -m main.ml.xgboost.train_xgboost

echo Training LightGBM
python -m main.ml.lightgbm.train_lightgbm_quantile

echo Training MLP
python -m main.ml.mlp.train_mlp

echo Training Probabilistic MLP
python -m main.ml.probabilistic_mlp.train_pmlp
