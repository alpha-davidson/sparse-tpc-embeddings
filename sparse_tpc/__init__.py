"""Sparse ResNet and PointNet-style embeddings of TPC data (Wheeler et al., arXiv:2608.21756)."""

import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
