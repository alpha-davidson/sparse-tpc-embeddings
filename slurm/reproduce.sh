#!/bin/bash
#SBATCH --job-name=sparse-tpc
#SBATCH --gpus=1
#SBATCH --mem=48g
#SBATCH --output=reproduce-%j.out
#
# Run the full pipeline of the paper: pretraining, the random baseline,
# embeddings, linear probes (Table 3) and PCA figures (Figs. 2 and 3).
#
# Usage (from the repository root, with the environment activated):
#   DATA_DIR=/path/to/data sbatch slurm/reproduce.sh
#
# Expected files (see "Data" in the README):
#   $DATA_DIR/gadget/run_0238_filtered_labeled_{train,val,test}_bands.npy
#   $DATA_DIR/gadget/gadget_3class.npy
#   $DATA_DIR/attpc/attpc_o16.npy
set -euo pipefail

: "${DATA_DIR:?set DATA_DIR to the data directory}"
OUT=${OUT:-runs/reproduce}
SEED=${SEED:-0}
G=$DATA_DIR/gadget
mkdir -p "$OUT/embeddings" "$OUT/figures"

python scripts/train.py --seed "$SEED" --out "$OUT/pretrain" \
    --train "$G/run_0238_filtered_labeled_train_bands.npy" \
    --val "$G/run_0238_filtered_labeled_val_bands.npy" \
    --test "$G/run_0238_filtered_labeled_test_bands.npy"
python scripts/make_random_model.py --seed "$SEED" --out "$OUT/random/random.pt"

for model in train rand; do
    weights=$OUT/pretrain/best.pt
    [ "$model" = rand ] && weights=$OUT/random/random.pt
    python scripts/extract_embeddings.py --weights "$weights" \
        --data "$G/gadget_3class.npy" --out "$OUT/embeddings/gadget_$model.npz"
    python scripts/extract_embeddings.py --weights "$weights" \
        --data "$DATA_DIR/attpc/attpc_o16.npy" --out "$OUT/embeddings/attpc_$model.npz"
done

E=$OUT/embeddings
python scripts/linear_probe.py ResNet_train="$E/gadget_train.npz" ResNet_rand="$E/gadget_rand.npz" \
    --out "$OUT/probe_gadget.json"
python scripts/linear_probe.py ResNet_train="$E/attpc_train.npz" ResNet_rand="$E/attpc_rand.npz" \
    --groups 0,1,2 3 4,5 --out "$OUT/probe_attpc.json"

python scripts/plot_pca.py gadget --rand "$E/gadget_rand.npz" --train "$E/gadget_train.npz" \
    --out "$OUT/figures/fig2_pca.pdf"
python scripts/plot_pca.py attpc --rand "$E/attpc_rand.npz" --train "$E/attpc_train.npz" \
    --out "$OUT/figures/fig3_attpc_pca.pdf"
