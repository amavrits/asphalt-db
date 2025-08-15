source .venv/bin/activate

n_splits=100
epochs=10000
lr=0.0001
bitumens=(True False)
#logys=(False True)
logys=(False)

for bitumen in "${bitumens[@]}"; do
    echo "Bitumen: $bitumen"
    for logy in "${logys[@]}"; do
      echo "Log Y: $logy"
      if [ "$bitumen" = True ]; then
        if [ "$logy" = True ]; then
          python -m main.ml.probabilistic_mlp.run_pmlp \
          --n_splits=$n_splits \
          --epochs=$epochs \
          --lr=$lr \
          --n_splits=$n_splits \
          --use_bitumen \
          --log_y
        else
          python -m main.ml.probabilistic_mlp.run_pmlp \
          --n_splits=$n_splits \
          --epochs=$epochs \
          --lr=$lr \
          --n_splits=$n_splits \
          --use_bitumen
        fi
      else
        if [ "$logy" = True ]; then
          python -m main.ml.probabilistic_mlp.run_pmlp \
          --n_splits=$n_splits \
          --epochs=$epochs \
          --lr=$lr \
          --n_splits=$n_splits \
          --log_y
        else
          python -m main.ml.probabilistic_mlp.run_pmlp \
          --n_splits=$n_splits \
          --epochs=$epochs \
          --lr=$lr \
          --n_splits=$n_splits
        fi
      fi
    done
done

