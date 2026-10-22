"""PCA views of the embeddings (Figures 2 and 3).

``gadget`` draws PCA1 vs PCA2 for ResNet_rand and ResNet_train side by side,
colored by class (Fig. 2, middle and right panels).

``attpc`` draws the AT-TPC layout of Fig. 3: PCA1 vs PCA2 scatter plots for
both models (a, b) and the distributions of PCA1 and PCA2 (c, d). Embeddings
are standardized before PCA, and only events with track-count labels 0-5
are kept.

Examples:
    python scripts/plot_pca.py gadget --rand emb/gadget_rand.npz --train emb/gadget_train.npz --out fig2_pca.pdf
    python scripts/plot_pca.py attpc --rand emb/attpc_rand.npz --train emb/attpc_train.npz --out fig3.pdf
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

GADGET_CLASSES = ["800 keV p", "1600 keV p", "2 MeV α"]


def project(path, standardize, keep_labels=None):
    data = np.load(path)
    X, y = data["embeddings"], data["labels"].astype(int)
    if keep_labels is not None:
        mask = np.isin(y, keep_labels)
        X, y = X[mask], y[mask]
    if standardize:
        X = StandardScaler().fit_transform(X)
    return PCA(n_components=2).fit_transform(X), y


def plot_gadget(args):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, name, path in zip(axes, ("ResNet_rand", "ResNet_train"), (args.rand, args.train)):
        X_pca, y = project(path, args.standardize)
        for label in np.unique(y):
            name_ = GADGET_CLASSES[label] if label < len(GADGET_CLASSES) else str(label)
            sel = y == label
            ax.scatter(X_pca[sel, 0], X_pca[sel, 1], s=6, alpha=0.6, label=name_)
        ax.set_title(name)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
    axes[1].legend(markerscale=2, fontsize=8)
    return fig


def plot_attpc(args):
    fig = plt.figure(figsize=(7.5, 4.2))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    ax_rand, ax_train = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_hist = [fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])]
    colors = plt.get_cmap("tab10").colors

    for ax, name, path, color in (
        (ax_rand, "ResNet_rand", args.rand, colors[0]),
        (ax_train, "ResNet_train", args.train, colors[3]),
    ):
        X_pca, y = project(path, standardize=True, keep_labels=list(range(6)))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=10, cmap="tab10", alpha=0.2)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        for pc, hist_ax in enumerate(ax_hist):
            hist_ax.hist(X_pca[:, pc], bins=100, range=(-50, 150), alpha=0.7, color=color, label=name)

    for pc, hist_ax in enumerate(ax_hist):
        hist_ax.set_xlabel(f"PCA{pc + 1}")
        hist_ax.set_ylabel("counts")
        hist_ax.grid(axis="y", alpha=0.5)
        hist_ax.legend()
    for ax, label in zip((ax_rand, ax_train, *ax_hist), ("a)", "b)", "c)", "d)")):
        ax.text(-0.08, 1.02, label, transform=ax.transAxes, fontsize=12, va="bottom", ha="left")
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["gadget", "attpc"])
    p.add_argument("--rand", required=True, help="ResNet_rand embeddings (.npz)")
    p.add_argument("--train", required=True, help="ResNet_train embeddings (.npz)")
    p.add_argument("--out", required=True, help="figure path (.pdf or .png)")
    p.add_argument("--standardize", action="store_true", help="standardize before PCA (gadget only)")
    args = p.parse_args()

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    fig = plot_gadget(args) if args.dataset == "gadget" else plot_attpc(args)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
