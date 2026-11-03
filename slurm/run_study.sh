#!/bin/bash
#SBATCH --job-name=tpc-study
#SBATCH --gpus=1
#SBATCH --mem=32g
#SBATCH --array=0-5
#SBATCH --output=study-%A_%a.out
#
# Run every encoder of the paper as a Slurm array job, one GPU per encoder:
#   0 ResNet / Mg20-2    1 ResNet / O16-2    2 ResNet / random
#   3 PointNet / Mg20-2  4 PointNet / O16-2  5 PointNet / random
# Each job trains (or initializes) five encoders and probes them on all four
# evaluation tasks. Then build the tables and figures:
#   python scripts/make_tables.py $OUT --compare
#   python scripts/plot_pca.py $OUT --eval O16-3 --fold 4 --out fig5.pdf
#   python scripts/plot_pca.py $OUT --eval Mg21-3 --fold 3 --out fig6.pdf
#
# Usage (from the repository root, with the environment activated):
#   DATA_DIR=/path/to/data OUT=runs/study sbatch slurm/run_study.sh
set -euo pipefail

: "${DATA_DIR:?set DATA_DIR to the folder holding the dataset folders}"
OUT=${OUT:-runs/study}
ARCHS=(resnet resnet resnet pointnet pointnet pointnet)
SOURCES=(Mg20-2 O16-2 random Mg20-2 O16-2 random)
i=${SLURM_ARRAY_TASK_ID:-0}

python scripts/run_study.py --arch "${ARCHS[$i]}" --source "${SOURCES[$i]}" \
    --data-root "$DATA_DIR" --out "$OUT" --save-embeddings O16-3 Mg21-3
