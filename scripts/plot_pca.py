"""PCA views of the embeddings (Figs. 5 and 6).

Draws a 2 x 2 grid comparing random and 16O-trained ResNet and PointNet-style
encoders on one evaluation task, using the embeddings saved by
``run_study.py --save-embeddings O16-3 Mg21-3``. The paper shows the O16-3 task from fold 4
(Fig. 5) and the Mg21-3 task from fold 3 (Fig. 6).

Examples:
    python scripts/plot_pca.py runs/study --eval O16-3 --fold 4 --out fig5.pdf
    python scripts/plot_pca.py runs/study --eval Mg21-3 --fold 3 --out fig6.pdf
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sparse_tpc.datasets import TASKS

COLORS = ["#0072B2", "#D55E00", "#009E73"]
PANELS = [
    ("resnet", "random", "Random ResNet"),
    ("resnet", "O16-2", r"$^{16}$O-trained ResNet"),
    ("pointnet", "random", "Random PointNet"),
    ("pointnet", "O16-2", r"$^{16}$O-trained PointNet"),
]


def project(path, standardize, n_components):
    data = np.load(path)
    X, y = data["embeddings"], data["labels"].astype(int)
    if standardize:
        X = StandardScaler().fit_transform(X)
    return PCA(n_components=n_components, svd_solver="randomized", random_state=28).fit_transform(X), y


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="the --out directory of run_study.py")
    p.add_argument("--eval", required=True, choices=sorted(TASKS), help="evaluation task")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--out", required=True, help="figure path (.pdf or .png)")
    p.add_argument("--dims", type=int, choices=[2, 3], default=3, help="principal components to show")
    p.add_argument("--no-standardize", action="store_true", help="skip standardizing embeddings before PCA")
    args = p.parse_args()

    class_names = TASKS[args.eval].class_names
    fig = plt.figure(figsize=(9.0, 8.2))
    for i, (arch, source, title) in enumerate(PANELS):
        path = Path(args.root, arch, source, "embeddings", f"{args.eval}_fold{args.fold}.npz")
        X, y = project(path, not args.no_standardize, args.dims)
        ax = fig.add_subplot(2, 2, i + 1, projection="3d" if args.dims == 3 else None)
        # Draw the largest class first so smaller classes stay visible.
        for label in np.argsort(np.bincount(y))[::-1]:
            pts = X[y == label].T
            ax.scatter(*pts, s=6, alpha=0.3, color=COLORS[label], linewidths=0,
                       rasterized=True, label=class_names[label])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if args.dims == 3:
            ax.set_zlabel("PC3")
            ax.view_init(elev=20, azim=-58)
        ax.set_title(f"({chr(97 + i)}) {title}", loc="left", fontsize=12)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    order = [labels.index(name) for name in class_names]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="upper center", ncol=len(order), frameon=False, markerscale=1.8)
    # The tight bounding box ignores 3D axis labels, so pad it to keep "PC3" in view.
    fig.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0.35, facecolor="white")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
