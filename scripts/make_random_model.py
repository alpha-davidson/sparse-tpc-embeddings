"""Save a randomly initialized sparse ResNet (the ResNet_rand baseline).

Example:
    python scripts/make_random_model.py --out runs/random/random.pt --seed 0
"""

import argparse
import os

import torch

from sparse_tpc import seed_everything
from sparse_tpc.resnet import MODELS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="path of the state dict to write")
    p.add_argument("--model", default="resnet14", choices=sorted(MODELS))
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    net = MODELS[args.model](in_channels=4, out_channels=args.num_classes, D=3)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print(f"saved randomly initialized {args.model} to {args.out}")


if __name__ == "__main__":
    main()
