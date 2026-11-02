"""Assemble Tables 3-5 (probe results) from run_study.py outputs.

Each cell is the mean ± standard deviation over the five folds of linear and
RBF probe accuracy and macro F1. With ``--compare`` the paper's means are
shown next to each row.

Example:
    python scripts/make_tables.py runs/study --compare > tables.md
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sparse_tpc.datasets import same_experiment

# (table title, [(evaluation task, encoder-training source), ...])
TABLES = [
    ("Table 3: within-experiment", [("Mg20-3", "Mg20-2"), ("O16-3", "O16-2")]),
    ("Table 4: cross-experiment", [("Mg21-3", "Mg20-2"), ("B10-2", "O16-2")]),
    ("Table 5: cross-detector", [("O16-3", "Mg20-2"), ("B10-2", "Mg20-2"),
                                 ("Mg20-3", "O16-2"), ("Mg21-3", "O16-2")]),
]
METRICS = [("linear", "accuracy"), ("linear", "macro_f1"), ("rbf", "accuracy"), ("rbf", "macro_f1")]

# Means reported in arXiv:2608.21756v2 (linear acc, linear F1, RBF acc, RBF F1).
PAPER = {
    ("Mg20-3", "naive"): (0.5428, 0.2346, 0.5428, 0.2346),
    ("Mg20-3", "random", "resnet"): (0.9109, 0.8840, 0.9321, 0.9077),
    ("Mg20-3", "random", "pointnet"): (0.9979, 0.9977, 0.9933, 0.9928),
    ("Mg20-3", "Mg20-2", "resnet"): (0.9782, 0.9732, 0.9803, 0.9753),
    ("Mg20-3", "Mg20-2", "pointnet"): (0.9999, 0.9999, 0.9999, 0.9999),
    ("Mg20-3", "O16-2", "resnet"): (0.9451, 0.9357, 0.9842, 0.9808),
    ("Mg20-3", "O16-2", "pointnet"): (0.9994, 0.9992, 0.9997, 0.9997),
    ("O16-3", "naive"): (0.4817, 0.2167, 0.4817, 0.2167),
    ("O16-3", "random", "resnet"): (0.6008, 0.5107, 0.6646, 0.4873),
    ("O16-3", "random", "pointnet"): (0.8008, 0.7579, 0.8390, 0.8044),
    ("O16-3", "O16-2", "resnet"): (0.7010, 0.6264, 0.7819, 0.6957),
    ("O16-3", "O16-2", "pointnet"): (0.8352, 0.7941, 0.8885, 0.8623),
    ("O16-3", "Mg20-2", "resnet"): (0.6955, 0.6162, 0.7817, 0.6960),
    ("O16-3", "Mg20-2", "pointnet"): (0.8248, 0.7841, 0.8720, 0.8489),
    ("Mg21-3", "naive"): (0.5632, 0.2402, 0.5632, 0.2402),
    ("Mg21-3", "random", "resnet"): (0.8524, 0.7985, 0.8499, 0.7922),
    ("Mg21-3", "random", "pointnet"): (0.9955, 0.9942, 0.9945, 0.9937),
    ("Mg21-3", "Mg20-2", "resnet"): (0.9506, 0.9246, 0.9622, 0.9422),
    ("Mg21-3", "Mg20-2", "pointnet"): (0.9967, 0.9958, 0.9985, 0.9983),
    ("Mg21-3", "O16-2", "resnet"): (0.8790, 0.8430, 0.9374, 0.9146),
    ("Mg21-3", "O16-2", "pointnet"): (0.9923, 0.9901, 0.9982, 0.9979),
    ("B10-2", "naive"): (0.5174, 0.3410, 0.5174, 0.3410),
    ("B10-2", "random", "resnet"): (0.5205, 0.5145, 0.5187, 0.4278),
    ("B10-2", "random", "pointnet"): (0.8723, 0.8722, 0.8965, 0.8963),
    ("B10-2", "O16-2", "resnet"): (0.6267, 0.6261, 0.6891, 0.6891),
    ("B10-2", "O16-2", "pointnet"): (0.8985, 0.8985, 0.9496, 0.9495),
    ("B10-2", "Mg20-2", "resnet"): (0.6426, 0.6420, 0.6885, 0.6883),
    ("B10-2", "Mg20-2", "pointnet"): (0.8889, 0.8888, 0.9355, 0.9354),
}


def load(root):
    runs = {}
    for path in Path(root).glob("*/*/results.json"):
        r = json.loads(path.read_text())
        runs[(r["arch"], r["source"])] = r["folds"]
    return runs


def cell(folds, evaluation, probe, metric):
    values = [f["eval"][evaluation][probe][metric] for f in folds if evaluation in f["eval"]]
    return (np.mean(values), np.std(values), len(values)) if values else None


def row(label, folds, evaluation, paper, probes=METRICS):
    cells = [cell(folds, evaluation, p, m) for p, m in probes] if folds else [None] * 4
    text = [f"{c[0]:.4f} ± {c[1]:.4f}" if c else "—" for c in cells]
    line = f"| {label} | " + " | ".join(text) + " |"
    if paper is not None:
        line += " " + " / ".join(f"{v:.4f}" for v in paper) + " |"
    return line


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="the --out directory of run_study.py")
    p.add_argument("--compare", action="store_true", help="add the paper's means")
    args = p.parse_args()
    runs = load(args.root)

    header = "| Encoder | Linear acc. | Linear macro-F1 | RBF acc. | RBF macro-F1 |"
    rule = "| --- | --- | --- | --- | --- |"
    if args.compare:
        header += " Paper |"
        rule += " --- |"
    for title, blocks in TABLES:
        print(f"\n### {title}\n")
        for evaluation, source in blocks:
            print(f"\n**{evaluation}** (trained encoders: {source})\n")
            print(header)
            print(rule)
            # Naive baseline: from the fold test sets when within-experiment,
            # otherwise from the 80/20 split (identical for every encoder).
            within = same_experiment(source, evaluation)
            naive_src = runs.get(("resnet", source if within else "random"))
            naive = [("naive", "accuracy"), ("naive", "macro_f1"), ("naive", "accuracy"), ("naive", "macro_f1")]
            paper = PAPER.get((evaluation, "naive")) if args.compare else None
            print(row("Naive baseline", naive_src, evaluation, paper, naive))
            for src, label in (("random", "Random"), (source, f"{source}-trained")):
                for arch, name in (("resnet", "ResNet"), ("pointnet", "PointNet")):
                    paper = PAPER.get((evaluation, src, arch)) if args.compare else None
                    print(row(f"{label} {name}", runs.get((arch, src)), evaluation, paper))


if __name__ == "__main__":
    main()
