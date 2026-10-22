"""Extract penultimate-layer embeddings (after global max pooling).

Writes an .npz file with ``embeddings`` (N, 512), ``labels`` and
``event_ids``. By default events are quantized directly from their absolute
positions, as in the embedding runs behind the paper; pass ``--shift-z`` to
use the pretraining convention (drift axis measured from the earliest hit).

Example:
    python scripts/extract_embeddings.py --weights runs/pretrain/best.pt \
        --data gadget_3class.npy --out embeddings/gadget_train.npz
"""

import argparse
import os

import numpy as np
import torch

from sparse_tpc.data import TPCEventDataset, make_loader, to_sparse
from sparse_tpc.resnet import MODELS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="state dict from train.py or make_random_model.py")
    p.add_argument("--data", required=True, help="events (.npy)")
    p.add_argument("--out", required=True, help="output .npz")
    p.add_argument("--model", default="resnet14", choices=sorted(MODELS))
    p.add_argument("--voxel-size", type=float, default=0.05)
    p.add_argument("--shift-z", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=1)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.weights, map_location=device)
    num_classes = state["final.linear.weight"].shape[0]
    net = MODELS[args.model](in_channels=4, out_channels=num_classes, D=3).to(device)
    net.load_state_dict(state)
    net.eval()

    dataset = TPCEventDataset(args.data, args.voxel_size, shift_z=args.shift_z)
    embeddings, labels, event_ids = [], [], []
    with torch.no_grad():
        for batch in make_loader(dataset, args.batch_size, args.num_workers):
            embeddings.append(net.embed(to_sparse(batch, device)).F.cpu().numpy())
            labels.append(batch["labels"].numpy())
            event_ids.append(batch["event_ids"].numpy())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    embeddings = np.concatenate(embeddings)
    np.savez_compressed(
        args.out,
        embeddings=embeddings,
        labels=np.concatenate(labels),
        event_ids=np.concatenate(event_ids),
    )
    print(f"saved {embeddings.shape[0]} embeddings of size {embeddings.shape[1]} to {args.out}")


if __name__ == "__main__":
    main()
