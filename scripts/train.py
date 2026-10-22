"""Pretrain the sparse ResNet on GADGET II proton/alpha classification.

Defaults follow Table 2 of the paper: cross-entropy loss, Adam (lr 5e-4,
weight decay 1e-4), cosine annealing with T_max = 13, batch size 64, gradient
clipping at max-norm 1.0, 15 epochs, a weighted random sampler, and the
checkpoint with the lowest validation loss kept as the final model.

Example:
    python scripts/train.py --train train.npy --val val.npy --test test.npy --out runs/pretrain
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from sparse_tpc import seed_everything
from sparse_tpc.data import TPCEventDataset, make_loader, to_sparse
from sparse_tpc.resnet import MODELS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="training events (.npy)")
    p.add_argument("--val", required=True, help="validation events (.npy)")
    p.add_argument("--test", help="test events (.npy), evaluated with the best checkpoint")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--model", default="resnet14", choices=sorted(MODELS))
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--voxel-size", type=float, default=0.05)
    p.add_argument("--shift-range", type=float, default=0.9, help="max random translation (mm) per axis")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--t-max", type=int, default=13, help="cosine annealing period (epochs)")
    p.add_argument("--no-weighted-sampler", action="store_true", help="shuffle uniformly instead")
    p.add_argument("--class-weights", type=float, nargs="+", help="per-class cross-entropy weights")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def evaluate(net, loader, criterion, device):
    net.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for batch in loader:
        out = net(to_sparse(batch, device)).F
        total_loss += criterion(out, batch["labels"].to(device)).item()
        y_true.extend(batch["labels"].tolist())
        y_pred.extend(out.argmax(1).cpu().tolist())
    return total_loss / len(loader), np.array(y_true), np.array(y_pred)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)

    train_set = TPCEventDataset(args.train, args.voxel_size, augment=True, shift_range=args.shift_range)
    val_set = TPCEventDataset(args.val, args.voxel_size)
    for name, ds in (("train", train_set), ("val", val_set)):
        print(f"{name}: {len(ds)} events, class counts {np.bincount(ds.labels).tolist()}")

    train_loader = make_loader(
        train_set,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        weighted=not args.no_weighted_sampler,
        generator=generator,
    )
    val_loader = make_loader(val_set, args.batch_size, args.num_workers)

    net = MODELS[args.model](in_channels=4, out_channels=args.num_classes, D=3).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
    weight = torch.tensor(args.class_weights, device=device) if args.class_weights else None
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    best_path = os.path.join(args.out, "best.pt")
    best_val_loss = float("inf")
    history = []
    for epoch in range(args.epochs):
        net.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = net(to_sparse(batch, device)).F
            loss = criterion(out, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss, y_true, y_pred = evaluate(net, val_loader, criterion, device)
        val_acc = accuracy_score(y_true, y_pred)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc})
        print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  val acc {val_acc:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), best_path)
        scheduler.step()

    torch.save(net.state_dict(), os.path.join(args.out, "last.pt"))
    results = {"args": vars(args), "history": history, "best_val_loss": best_val_loss}

    if args.test:
        test_loader = make_loader(TPCEventDataset(args.test, args.voxel_size), args.batch_size, args.num_workers)
        net.load_state_dict(torch.load(best_path, map_location=device))
        _, y_true, y_pred = evaluate(net, test_loader, criterion, device)
        print(classification_report(y_true, y_pred, digits=3))
        results["test"] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "report": classification_report(y_true, y_pred, output_dict=True),
        }
        print(f"test accuracy {results['test']['accuracy']:.4f}  macro F1 {results['test']['macro_f1']:.4f}")

    with open(os.path.join(args.out, "train_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
