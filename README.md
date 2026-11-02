# sparse-tpc-embeddings

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23889819.svg)](https://doi.org/10.5281/zenodo.23889819)

Sparse ResNet and PointNet-style encoders for time projection chamber (TPC) events, built on [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) sparse tensors, with the probes used to test whether their embeddings carry over across tasks, experiments and detectors. This is the code for:

> T. Wheeler, M. P. Kuchera, R. Ramanujan, W. Sieland, R. Krupp, Y. Ayyad, D. Bazin, C. L. Cross, H. Y. I. Heung, A. J. Jones, R. Mahajan, S. Ravishankar, P. Singh, B. Votaw, and C. Wrede,
> **How Architecture and Training Affect TPC Representations Across Experiments**,
> [arXiv:2608.21756](https://arxiv.org/abs/2608.21756) (2026).

The code for the earlier NeurIPS 2025 workshop paper, [Sparse Methods for Vector Embeddings of TPC Data](https://arxiv.org/abs/2511.11221) (arXiv:2511.11221), is release [v1.0.0](https://github.com/alpha-davidson/sparse-tpc-embeddings/tree/v1.0.0) ([doi:10.5281/zenodo.23889820](https://doi.org/10.5281/zenodo.23889820)). Check out that tag to reproduce the workshop paper:

```bash
git checkout v1.0.0
```

## Overview

Each encoder maps a variable-length TPC event (hit positions and charges) to a 512-dimensional embedding:

* **ResNet14**: a sparse convolutional ResNet over a 0.05 mm voxel grid.
* **PointNet-style encoder**: a shared per-point MLP followed by global max pooling, implemented with Minkowski Engine sparse tensors so events keep their variable length.

Each architecture is either trained on a simple source task (²⁰Mg proton vs. alpha, or ¹⁶O 0–2 vs. 3–5 tracks) or left randomly initialized. It is then frozen, and linear and RBF support vector machines (SVMs) are trained on its embeddings for downstream tasks. These tasks fall into three groups: within the same experiment (Table 3), in another experiment on the same detector (Table 4), and on the other detector (Table 5). Every encoder is trained five times with five-fold cross-validation.

## Contents

| Path | Purpose | Paper |
| --- | --- | --- |
| `sparse_tpc/datasets.py` | Tasks and labels, five-fold splits, input scaling | §III.1, Table 2 |
| `sparse_tpc/resnet.py`, `sparse_tpc/pointnet.py` | The two encoders | §III.2, Figs. 3–4 |
| `sparse_tpc/encoders.py` | Turns events into network inputs for each architecture | §III.2 |
| `sparse_tpc/training.py` | Source-task training | Appendix A, Table 6 |
| `sparse_tpc/probes.py` | Embedding extraction, linear and RBF SVM probes, naive baseline | §III.3 |
| `scripts/run_study.py` | Trains or initializes one encoder in five folds and probes all tasks | Tables 3–5 |
| `scripts/make_tables.py` | Builds Tables 3–5 from the results, optionally next to the paper's values | Tables 3–5 |
| `scripts/plot_pca.py` | PCA views of the embeddings | Figs. 5–6 |
| `slurm/run_study.sh` | Runs all six encoders as a Slurm array job | |

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

The data come from four FRIB experiments (paper Table 2):

| Folder | Detector | Experiment | Tasks | Events |
| --- | --- | --- | --- | --- |
| `Mg20-3_filtered_full_variable` | GADGET II | E21072, ²⁰Mg β-delayed decays | `Mg20-2` (proton / alpha, source task), `Mg20-3` (770 keV p / 1596 keV p / 2153 keV α) | 34,579 |
| `Mg21-3_filtered_full_variable` | GADGET II | E25058, ²¹Mg β-delayed decays | `Mg21-3` (1285 keV p / 1930 keV p / 2153 keV α) | 5,975 |
| `O16-3_filtered_full_variable` | AT-TPC | E20020, ¹⁶O + α | `O16-2` (0–2 / 3–5 tracks, source task), `O16-3` (0–2 / 3 / 4–5 tracks) | 2,458 |
| `B10-2_filtered_full_variable` | AT-TPC | E20009, ¹⁰B + d | `B10-2` (3 / 4 tracks) | 4,461 |

Each folder holds one `<event number>.npy` file per event: an (N, 7) array with one row per hit and the columns x, y, z (mm), charge, event number, raw label (decay branch or number of tracks), and number of hits. Put the four folders in one directory and pass it as `--data-root` (or `DATA_DIR`).

The processed data will be made publicly available when the paper is published, in accordance with the FRIB Data Management and Sharing Plan. Access to additional FRIB experimental data is subject to the applicable data-sharing policies and approval of the experiment co-spokespersons.

## Reproducing the paper

Run all six encoders (ResNet and PointNet, each trained on ²⁰Mg, trained on ¹⁶O, or random) as one Slurm array job:

```bash
DATA_DIR=/path/to/data OUT=runs/study sbatch slurm/run_study.sh
```

Or run them one at a time:

```bash
for arch in resnet pointnet; do
  for source in Mg20-2 O16-2 random; do
    python scripts/run_study.py --arch $arch --source $source --data-root $DATA_DIR --out runs/study \
        --save-embeddings O16-3 Mg21-3
  done
done
```

Then build the tables and figures:

```bash
python scripts/make_tables.py runs/study --compare     # Tables 3-5, with the paper's means alongside
python scripts/plot_pca.py runs/study --eval O16-3 --fold 4 --out fig5.pdf
python scripts/plot_pca.py runs/study --eval Mg21-3 --fold 3 --out fig6.pdf
```

`run_study.py` defaults to the paper's settings (Table 6):
* AdamW with weight decay 10⁻⁴ and batch size 128.
* `ReduceLROnPlateau` on the validation loss: factor 0.1, patience 10, relative threshold 10⁻³, minimum learning rate 5 × 10⁻⁸.
* At most 200 epochs, with early stopping after 30 epochs without improvement. The checkpoint with the lowest validation loss is kept.
* A weighted training sampler.
* Initial learning rate 5 × 10⁻⁴, except 5 × 10⁻⁶ for the PointNet-style encoder trained on ²⁰Mg.
* ResNet14 only: gradient clipping at norm 1.0, and a random translation of each training event by up to 0.9 mm per axis.
* Fold *i* uses split seed 28 and training seed 28 + *i*.

On one GPU, the ²⁰Mg-trained ResNet14 takes about 6 hours. The other five encoders take 15–60 minutes each.

### Expected results

RESULTS_TBD

### Notes

* Training on a GPU is not bit-for-bit deterministic, so retrained encoders give slightly different numbers from the paper. Random encoders are reproduced exactly from their seeds. With the paper's saved fold-0 checkpoints, this code reproduces the paper's probe results exactly for 15 of 24 encoder–task pairs. The rest differ by at most 0.004 (one or two test events), which we have not traced further.
* Input scaling (charge range and the GADGET II drift-axis standardization) is computed only from training and validation events, never from the events a probe is tested on.

## Acknowledgements

We thank the GADGET and AT-TPC collaborations for their efforts in collecting and providing the experimental data used in this work.

This work was supported by the U.S. Department of Energy, Office of Science, under award No. DE-SC0024587, and by the U.S. National Science Foundation award [OAC-2311263](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2311263). This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Nuclear Physics and used resources of the Facility for Rare Isotope Beams (FRIB) Operations, which is a DOE Office of Science User Facility under Award Number DE-SC0023633.

The five-fold training and probing pipelines were written by Liam Sieland and refined by Michelle P. Kuchera for the paper's final runs. They build on the ALPhA group's FRIBMink sparse-convolution code, written by Liam Sieland, Connor L. Cross, Raghuram Ramanujan, Michelle P. Kuchera, Benjamin Votaw, Andrew J. Jones and Ryan Krupp; on the GADGET II pretraining and embedding code by Ryan Krupp and Tyler Wheeler; and on the ResNet and PointNet examples in the [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) (Choy et al., 2019).

## Citation

If you use this code, please cite the paper. The code itself is archived on Zenodo as [doi:10.5281/zenodo.23889819](https://doi.org/10.5281/zenodo.23889819).

```bibtex
@article{wheeler2026architecture,
  title   = {How Architecture and Training Affect {TPC} Representations Across Experiments},
  author  = {Wheeler, Tyler and Kuchera, Michelle P. and Ramanujan, Raghuram and Sieland, William and
             Krupp, Ryan and Ayyad, Yassid and Bazin, Daniel and Cross, Connor L. and Heung, Hoi Yan Ian and
             Jones, Andrew J. and Mahajan, Ruchi and Ravishankar, Saiprasad and Singh, Pranjal and
             Votaw, Benjamin and Wrede, Chris},
  journal = {arXiv preprint arXiv:2608.21756},
  year    = {2026},
  doi     = {10.48550/arXiv.2608.21756}
}
```

For the NeurIPS 2025 workshop paper (code in release v1.0.0):

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
