# Sparse Methods for Vector Embeddings of TPC Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23889819.svg)](https://doi.org/10.5281/zenodo.23889819)

Code for the paper

> T. Wheeler, M. P. Kuchera, R. Ramanujan, R. Krupp, C. Wrede, S. Ravishankar, C. L. Cross, H. Y. I. Heung, A. J. Jones, and B. Votaw,
> **Sparse Methods for Vector Embeddings of TPC Data**,
> Machine Learning and the Physical Sciences Workshop, NeurIPS 2025.
> [arXiv:2511.11221](https://arxiv.org/abs/2511.11221)

A sparse ResNet14 built with the [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) is pretrained to classify protons and alphas in GADGET II time projection chamber (TPC) events. Its penultimate-layer embeddings are then probed with a linear SVM and visualized with PCA. The probes use two tasks: a three-class GADGET II task, and track counting in a different detector, the Active-Target TPC (AT-TPC). A randomly initialized copy of the same network (ResNet_rand) serves as a baseline.

## Contents

| Path | Purpose | Paper |
| --- | --- | --- |
| `sparse_tpc/resnet.py` | Sparse ResNet14 (and ResNet50) with an `embed()` method | Table 1 |
| `sparse_tpc/data.py` | Event loading, random translation, voxelization, batching | §2.1 |
| `scripts/train.py` | Pretraining on GADGET II proton/alpha classification (ResNet_train) | §2.2, Table 2 |
| `scripts/make_random_model.py` | Randomly initialized network (ResNet_rand) | §2.3 |
| `scripts/prepare_attpc.py` | Converts labeled AT-TPC ¹⁶O + α events to the GADGET II format | §2.3 |
| `scripts/extract_embeddings.py` | 512-dimensional embeddings after global max pooling | §2.3 |
| `scripts/linear_probe.py` | Linear SVM probes and the naive baseline | Table 3 |
| `scripts/plot_pca.py` | PCA plots | Figs. 2 and 3 |
| `slurm/reproduce.sh` | Runs the whole pipeline as one Slurm job | |

## Installation

The code needs Linux and an NVIDIA GPU. It was tested with Python 3.8, PyTorch 1.9.0 and CUDA 11.1, Minkowski Engine 0.5.4, NumPy 1.21.6, scikit-learn 1.3.0, SciPy 1.10.1, matplotlib 3.7.2 and pandas 1.5.3.

```bash
git clone https://github.com/alpha-davidson/sparse-tpc-embeddings.git
cd sparse-tpc-embeddings
conda env create -f environment.yml
conda activate sparse-tpc
# Build the Minkowski Engine against the environment's PyTorch. This needs a
# CUDA 11.1 toolkit (nvcc) and a compatible GCC (we used GCC 10.2) on the build machine.
pip install MinkowskiEngine==0.5.4 --no-deps --no-build-isolation --config-settings="--blas=openblas"
pip install -e . --no-deps
python -c "import torch, MinkowskiEngine as ME; print(torch.cuda.is_available(), ME.__version__)"
```

See the [Minkowski Engine installation notes](https://github.com/NVIDIA/MinkowskiEngine#installation) if the build fails.

## Data

The experimental data are not distributed with this repository. They come from:

* **GADGET II** at FRIB, experiment e21072 (run 238). Events were labeled as protons or alphas by gating bands in range versus energy (overlap region excluded), using the [GADGET II analysis GUI](https://github.com/GADGET-group/gadget_data_analysis). The three-class probe set holds 212 events each of 800 keV protons, 1600 keV protons and 2 MeV alphas.
* **AT-TPC**, a ¹⁶O + α experiment. Events were hand-labeled with their number of tracks.

Please contact the authors about access. Every dataset is a pickled NumPy object array with one row per event:

```python
(points, label, event_id)   # points: (N, 4) float array with columns x, y, z [mm] and charge
```

`slurm/reproduce.sh` expects this layout:

```
$DATA_DIR/gadget/run_0238_filtered_labeled_{train,val,test}_bands.npy   # pretraining: label 0 = alpha, 1 = proton
$DATA_DIR/gadget/gadget_3class.npy                                     # probe: 0 = 800 keV p, 1 = 1600 keV p, 2 = 2 MeV alpha
$DATA_DIR/attpc/attpc_o16.npy                                           # probe: label = number of tracks
```

Create `attpc_o16.npy` from the zero-padded AT-TPC hit array and the track-count labels:

```bash
python scripts/prepare_attpc.py --events O16_w_event_keys.npy --labels O16_labels.csv --out $DATA_DIR/attpc/attpc_o16.npy
```

## Reproducing the paper

Run everything as one job:

```bash
DATA_DIR=/path/to/data sbatch slurm/reproduce.sh
```

Or step by step:

```bash
# 1. Pretrain ResNet_train (Table 2 settings are the defaults) and make ResNet_rand
python scripts/train.py --train $G/run_0238_filtered_labeled_train_bands.npy \
    --val $G/run_0238_filtered_labeled_val_bands.npy --test $G/run_0238_filtered_labeled_test_bands.npy \
    --out runs/pretrain
python scripts/make_random_model.py --out runs/random/random.pt

# 2. Embeddings for both probe datasets and both models
python scripts/extract_embeddings.py --weights runs/pretrain/best.pt --data $G/gadget_3class.npy --out emb/gadget_train.npz
python scripts/extract_embeddings.py --weights runs/random/random.pt --data $G/gadget_3class.npy --out emb/gadget_rand.npz
python scripts/extract_embeddings.py --weights runs/pretrain/best.pt --data $DATA_DIR/attpc/attpc_o16.npy --out emb/attpc_train.npz
python scripts/extract_embeddings.py --weights runs/random/random.pt --data $DATA_DIR/attpc/attpc_o16.npy --out emb/attpc_rand.npz

# 3. Linear probes (Table 3); AT-TPC track counts are grouped as {0,1,2}, {3}, {4,5}
python scripts/linear_probe.py ResNet_train=emb/gadget_train.npz ResNet_rand=emb/gadget_rand.npz
python scripts/linear_probe.py ResNet_train=emb/attpc_train.npz ResNet_rand=emb/attpc_rand.npz --groups 0,1,2 3 4,5

# 4. PCA figures
python scripts/plot_pca.py gadget --rand emb/gadget_rand.npz --train emb/gadget_train.npz --out fig2_pca.pdf
python scripts/plot_pca.py attpc --rand emb/attpc_rand.npz --train emb/attpc_train.npz --out fig3_attpc_pca.pdf
```

The whole pipeline takes about 25 minutes on one NVIDIA RTX 2080 Ti (about 17 minutes on an RTX A6000). Nearly all of that time is pretraining.

### Expected results

We reran the full pipeline with this code. The table compares the paper's Table 3 with the rerun, which used three pretraining seeds (`SEED=0,1,2`) and five probe splits per seed. Each entry gives the mean and, in brackets, the range of the per-seed means. Pretraining reached 0.994 test accuracy and macro F1 for all three seeds (paper: about 0.995 and 0.99).

| Probe | Metric | ResNet_train (paper) | ResNet_train (rerun) | ResNet_rand (paper) | ResNet_rand (rerun) | Naive (paper) | Naive (rerun) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GADGET II, 3 classes | accuracy | 0.97 | 0.950 [0.939–0.958] | 0.85 | 0.792 [0.783–0.798] | 0.33 | 0.328 |
| | macro F1 | 0.97 | 0.949 [0.939–0.958] | 0.85 | 0.790 [0.782–0.797] | 0.17 | 0.165 |
| AT-TPC track count | accuracy | 0.74 | 0.739 [0.730–0.747] | 0.58 | 0.588 [0.574–0.601] | 0.48 | 0.482 |
| | weighted F1 | 0.70 | 0.705 [0.694–0.716] | 0.53 | 0.546 [0.532–0.563] | 0.31 | 0.313 |

The AT-TPC results and the naive baselines match the paper. On GADGET II, both models score a few points below the paper's single split. The paper's ResNet_rand number comes from one random initialization and one 80/20 split of 636 events, and the per-split standard deviation is 0.03–0.05 for that model. The conclusions of the paper hold in every run: pretraining improves the embeddings on both detectors.

### Notes on the metrics and the original code

* In Table 3 of the paper, the GADGET II F1 scores are **macro** averages and the AT-TPC F1 scores are **weighted** averages over classes. `linear_probe.py` reports accuracy, macro F1 and weighted F1 for every model, so both can be compared.
* The probes in the paper used a single 80/20 split. `linear_probe.py` repeats the split over several seeds (`--seeds`, default 0–4) and reports the mean and standard deviation. With about 500 AT-TPC test events, scores vary by a few percent between splits.
* This code is a cleaned-up version of the scripts used for the paper. The changes:
  * Random translation is applied only to training events. The original scripts applied it during validation and testing as well.
  * Paths and seeds are command-line options.
  * The weighted random sampler and 15 epochs from Table 2 are the defaults. The original training script used a class-weighted loss (`--class-weights 0.5 1.5 --no-weighted-sampler`) and a configurable number of epochs.

## Acknowledgements

This work was supported by the U.S. Department of Energy, Office of Science, under award No. DE-SC0024587 and the National Science Foundation Office of Advanced Cyberinfrastructure award No. [OAC-2311263](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2311263).

The code was written by Ryan Krupp and Tyler Wheeler (pretraining, embeddings and probes on GADGET II data) and Michelle P. Kuchera (AT-TPC conversion, probes and figures). It builds on the ALPhA group's FRIBMink sparse-convolution code, written by Liam Sieland, Connor L. Cross, Raghuram Ramanujan, Michelle P. Kuchera, Benjamin Votaw and Ryan Krupp, and on the ResNet example in the [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) (Choy et al., 2019).

## Citation

If you use this code, please cite the paper. The code itself is archived on Zenodo as [doi:10.5281/zenodo.23889819](https://doi.org/10.5281/zenodo.23889819).

The paper:

```bibtex
@inproceedings{wheeler2025sparse,
  title     = {Sparse Methods for Vector Embeddings of {TPC} Data},
  author    = {Wheeler, Tyler and Kuchera, Michelle P. and Ramanujan, Raghuram and Krupp, Ryan and
               Wrede, Chris and Ravishankar, Saiprasad and Cross, Connor L. and Heung, Hoi Yan Ian and
               Jones, Andrew J. and Votaw, Benjamin},
  booktitle = {Machine Learning and the Physical Sciences Workshop, NeurIPS},
  year      = {2025},
  eprint    = {2511.11221},
  archivePrefix = {arXiv},
  doi       = {10.48550/arXiv.2511.11221}
}
```

## License

[MIT](LICENSE)
