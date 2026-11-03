"""Train (or randomly initialize) one encoder in five folds and probe it on every task.

For each fold, the encoder is trained on its source task with that fold held
out, or randomly initialized for ``--source random``. It is then frozen, and
linear and RBF probes are fit on its embeddings of every evaluation task:

* within-experiment (evaluation task uses the same events as the source task,
  e.g. O16-2 -> O16-3): the probe trains on the fold's training + validation
  events and is tested on the fold's held-out events;
* otherwise: a stratified 80/20 split of the evaluation dataset, seeded per
  fold, with input scaling computed from the 80% part.

Results go to ``<out>/<arch>/<source>/results.json``; ``scripts/make_tables.py``
turns them into Tables 3-5.

Example:
    python scripts/run_study.py --arch resnet --source O16-2 --data-root data --out runs/study
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from sparse_tpc import seed_everything
from sparse_tpc.datasets import TASKS, kfold_split, load_folder, same_experiment, scaling_constants
from sparse_tpc.encoders import EventEncoder
from sparse_tpc.probes import extract, probe
from sparse_tpc.training import TrainConfig, evaluate, train_encoder

EVAL_TASKS = ["Mg20-3", "Mg21-3", "O16-3", "B10-2"]

# Table 6: architecture-specific settings.
DEFAULT_LR = {("resnet", "Mg20-2"): 5e-4, ("resnet", "O16-2"): 5e-4,
              ("pointnet", "Mg20-2"): 5e-6, ("pointnet", "O16-2"): 5e-4}
ARCH_SETTINGS = {
    "resnet": dict(grad_clip=1.0, translation_range=0.9, drop_last=False),
    "pointnet": dict(grad_clip=0.0, translation_range=0.0, drop_last=True),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=["resnet", "pointnet"])
    p.add_argument("--source", required=True, choices=["Mg20-2", "O16-2", "random"])
    p.add_argument("--eval", nargs="+", default=EVAL_TASKS, choices=EVAL_TASKS, help="evaluation tasks")
    p.add_argument("--data-root", required=True, help="folder containing the dataset folders")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--k", type=int, default=5, help="number of folds")
    p.add_argument("--folds", type=int, nargs="+", help="run only these folds (default: all)")
    p.add_argument("--seed", type=int, default=28, help="split seed; fold i also seeds training with seed + i")
    p.add_argument("--lr", type=float, help="initial learning rate (default: Table 6)")
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--save-embeddings", nargs="*", default=[], choices=EVAL_TASKS, metavar="TASK",
                   help="save the embeddings of these evaluation tasks (.npz), e.g. for plot_pca.py")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out, args.arch, args.source)
    (out / "models").mkdir(parents=True, exist_ok=True)
    trained = args.source != "random"

    settings = ARCH_SETTINGS[args.arch]
    cfg = TrainConfig(
        lr=args.lr or DEFAULT_LR.get((args.arch, args.source), 5e-4),
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        grad_clip=settings["grad_clip"],
        drop_last=settings["drop_last"],
        num_workers=args.num_workers,
    )
    cache = {}

    def events_of(task):
        folder = TASKS[task].folder
        if folder not in cache:
            cache[folder] = load_folder(args.data_root, folder)
            log(f"loaded {len(cache[folder])} events from {folder}")
        return cache[folder]

    results = {"arch": args.arch, "source": args.source, "k": args.k, "seed": args.seed,
               "train_config": vars(cfg) if trained else None, "folds": []}
    for fold in args.folds if args.folds is not None else range(args.k):
        seed_everything(args.seed + fold)
        log(f"===== {args.arch} / {args.source}: fold {fold} =====")
        fold_result = {"fold": fold, "eval": {}}

        if trained:
            src = TASKS[args.source]
            train, val, test = kfold_split(events_of(args.source), fold, args.k, seed=args.seed)
            src_scaling = scaling_constants(train + val, src.detector)
            encoder = EventEncoder(args.arch, src.num_classes, device, settings["translation_range"])
            fold_result["training"] = train_encoder(
                encoder, src, train, val, src_scaling, cfg, out / "models" / f"fold{fold}.pt", log=log
            )
            fold_result["source_test_accuracy"] = evaluate(encoder, src, test, src_scaling, cfg)
            log(f"source-task test accuracy {fold_result['source_test_accuracy']:.4f}")
        else:
            encoder = EventEncoder(args.arch, 2, device)
            torch.save(encoder.net.state_dict(), out / "models" / f"fold{fold}.pt")

        for name in args.eval:
            task = TASKS[name]
            if trained and same_experiment(args.source, name):
                scaling = scaling_constants(train + val, task.detector)
                X_tr, y_tr = extract(encoder, train + val, task, scaling, args.batch_size)
                X_te, y_te = extract(encoder, test, task, scaling, args.batch_size)
            else:
                events = events_of(name)
                labels = np.array([task.label(e.raw_label) for e in events])
                tr_idx, te_idx = train_test_split(
                    np.arange(len(events)), test_size=0.2, stratify=labels, random_state=args.seed + fold
                )
                scaling = scaling_constants([events[i] for i in tr_idx], task.detector)
                X, y = extract(encoder, events, task, scaling, args.batch_size)
                X_tr, y_tr, X_te, y_te = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]

            fold_result["eval"][name] = probe(X_tr, y_tr, X_te, y_te)
            r = fold_result["eval"][name]
            log(f"{name}: linear acc {r['linear']['accuracy']:.4f} F1 {r['linear']['macro_f1']:.4f} | "
                f"RBF acc {r['rbf']['accuracy']:.4f} F1 {r['rbf']['macro_f1']:.4f}")
            if name in args.save_embeddings:
                (out / "embeddings").mkdir(exist_ok=True)
                np.savez_compressed(
                    out / "embeddings" / f"{name}_fold{fold}.npz",
                    embeddings=np.concatenate([X_tr, X_te]),
                    labels=np.concatenate([y_tr, y_te]),
                    split=np.array(["train"] * len(y_tr) + ["test"] * len(y_te)),
                )

        results["folds"].append(fold_result)
        with open(out / "results.json", "w") as f:
            json.dump(results, f, indent=2)
    log(f"results written to {out / 'results.json'}")


if __name__ == "__main__":
    main()
