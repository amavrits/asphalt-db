# --- Activate venv ---
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "Warning: .venv/bin/activate not found; continuing without venv."
fi


# Delete result path (pass a directory to delete as the first arg).
DIR="${1:-results/ml/xgboost}"
if [ -n "$DIR" ]; then
  if [ -d "$DIR" ]; then
    echo "Deleting directory: $DIR"
    rm -rf "$DIR"
  else
    echo "Directory does not exist: $DIR"
  fi
else
  echo "No directory provided to delete. Skipping cleanup."
fi


n_splits=100
epochs=10_000
lr=0.0001
bitumens=(True False)
logys=(False True)
#bitumens=(False)
#logys=(False)

python -m main.ml.prepare_data

for bitumen in "${bitumens[@]}"; do
    echo "Bitumen: $bitumen"
    for logy in "${logys[@]}"; do
      echo "Log Y: $logy"
      if [ "$bitumen" = True ]; then
        if [ "$logy" = True ]; then
          python -m main.ml.xgboost.run_xgboost \
          --n_splits=$n_splits \
          --use_bitumen \
          --log_y
        else
          python -m main.ml.xgboost.run_xgboost \
          --n_splits=$n_splits \
          --use_bitumen
        fi
      else
        if [ "$logy" = True ]; then
          python -m main.ml.xgboost.run_xgboost \
          --n_splits=$n_splits \
          --log_y
        else
          python -m main.ml.xgboost.run_xgboost \
          --n_splits=$n_splits
        fi
      fi
    done
done

