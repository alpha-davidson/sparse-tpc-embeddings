"""Linear-probe embeddings with a linear SVM (Table 3).

Each embedding file is split 80/20 (stratified) into train and test sets, a
linear SVM (C = 1) is fit on the training embeddings, and accuracy, macro F1
and weighted F1 are reported on the test set, averaged over ``--seeds``.
A naive baseline that always predicts the most common training class is
reported alongside.

``--groups`` merges and filters labels before probing. The AT-TPC
track-counting task in the paper uses ``--groups 0,1,2 3 4,5``.

Examples:
    python scripts/linear_probe.py ResNet_train=emb/gadget_train.npz ResNet_rand=emb/gadget_rand.npz
    python scripts/linear_probe.py ResNet_train=emb/attpc_train.npz ResNet_rand=emb/attpc_rand.npz \
        --groups 0,1,2 3 4,5
"""

import argparse
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

METRICS = ("accuracy", "macro_f1", "weighted_f1")


def regroup(labels, groups):
    """Map each group of original labels to a new class index; drop the rest."""
    new = np.full(labels.shape, -1)
    for i, group in enumerate(groups):
        new[np.isin(labels, group)] = i
    return new >= 0, new


def scores(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def probe(X, y, seed, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    svm = SVC(kernel="linear", C=1.0).fit(X_train, y_train)
    majority = np.bincount(y_train).argmax()
    return scores(y_test, svm.predict(X_test)), scores(y_test, np.full_like(y_test, majority))


def summarize(runs):
    return {m: (float(np.mean([r[m] for r in runs])), float(np.std([r[m] for r in runs]))) for m in METRICS}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("embeddings", nargs="+", help="NAME=path.npz pairs from extract_embeddings.py")
    p.add_argument("--groups", nargs="+", help="comma-separated label groups, e.g. 0,1,2 3 4,5")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--out", help="write results as JSON")
    args = p.parse_args()

    groups = [[int(v) for v in g.split(",")] for g in args.groups] if args.groups else None
    results, naive = {}, []
    for item in args.embeddings:
        name, path = item.split("=", 1)
        data = np.load(path)
        X, y = data["embeddings"], data["labels"].astype(int)
        if groups:
            keep, y = regroup(y, groups)
            X, y = X[keep], y[keep]
        print(f"{name}: {len(y)} events, class counts {np.bincount(y).tolist()}")
        runs = [probe(X, y, seed, args.test_size) for seed in args.seeds]
        results[name] = summarize([r[0] for r in runs])
        naive = [r[1] for r in runs]
    results["Naive"] = summarize(naive)

    print(f"\nLinear SVM, {len(args.seeds)} seeds (mean ± std)")
    print(f"{'model':<14}" + "".join(f"{m:>20}" for m in METRICS))
    for name, res in results.items():
        print(f"{name:<14}" + "".join(f"{res[m][0]:>13.3f} ± {res[m][1]:.3f}" for m in METRICS))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"seeds": args.seeds, "groups": groups, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
