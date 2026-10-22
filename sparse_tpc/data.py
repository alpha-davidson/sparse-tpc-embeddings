"""Loading TPC point clouds for the Minkowski Engine.

Every dataset in this project is stored as a pickled NumPy object array with
one row per event::

    (points, label, event_id)

where ``points`` is an (N, 4) float array of hits with columns
``x, y, z, q`` (positions in mm, q the charge amplitude), ``label`` is an
integer class, and ``event_id`` is an integer event number. Load files with
``np.load(path, allow_pickle=True)``. Only load pickled files you trust.
"""

import sys

import numpy as np
import torch
import MinkowskiEngine as ME
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def _alias_numpy2_modules():
    # Files pickled with NumPy 2 refer to numpy._core, which NumPy 1.x
    # (required by the Minkowski Engine) calls numpy.core.
    if "numpy._core" not in sys.modules and not hasattr(np, "_core"):
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.multiarray"] = np.core.multiarray


def load_events(path):
    """Load an object array of (points, label, event_id) and drop empty events."""
    _alias_numpy2_modules()
    events = np.load(path, allow_pickle=True)
    return [
        (np.asarray(points, dtype=np.float64), int(label), int(event_id))
        for points, label, event_id in events
        if len(points) > 0
    ]


def voxelize(xyz, voxel_size, shift_z=True):
    """Quantize hit positions to integer voxel coordinates.

    With ``shift_z`` the drift coordinate is measured from the earliest hit
    in the event, as in the pretraining data loader.
    """
    xyz = np.array(xyz, dtype=np.float64)
    if shift_z:
        xyz[:, 2] -= xyz[:, 2].min()
    return np.floor(xyz / voxel_size).astype(np.int32)


class TPCEventDataset(Dataset):
    """Point clouds with voxel coordinates and (x, y, z, q) features.

    During pretraining each event gets a random translation of up to
    ``shift_range`` mm per axis (to make the model robust to where a track
    lands on the pad plane), its z axis is shifted to start at zero, and it is
    quantized with ``voxel_size``. The features are the translated positions
    and the raw charge.
    """

    def __init__(self, path, voxel_size=0.05, augment=False, shift_range=0.9, shift_z=True):
        self.events = load_events(path)
        self.voxel_size = voxel_size
        self.augment = augment
        self.shift_range = shift_range
        self.shift_z = shift_z

    def __len__(self):
        return len(self.events)

    @property
    def labels(self):
        return np.array([label for _, label, _ in self.events])

    def __getitem__(self, idx):
        points, label, event_id = self.events[idx]
        xyz = points[:, :3]
        if self.augment:
            xyz = xyz + np.random.uniform(-self.shift_range, self.shift_range, size=(1, 3))
        coords = voxelize(xyz, self.voxel_size, shift_z=self.shift_z)
        feats = np.hstack((xyz, points[:, 3:4]))
        return coords, feats, label, event_id


def collate(batch):
    coords, feats, labels, event_ids = zip(*batch)
    return {
        "coords": ME.utils.batched_coordinates(coords),
        "feats": torch.from_numpy(np.vstack(feats)).float(),
        "labels": torch.LongTensor(labels),
        "event_ids": torch.LongTensor(event_ids),
    }


def seed_worker(worker_id):
    # Give each worker (and each epoch) its own NumPy stream for augmentation.
    np.random.seed(torch.initial_seed() % 2**32)


def make_loader(dataset, batch_size=64, num_workers=1, shuffle=False, weighted=False, generator=None):
    """Build a DataLoader; ``weighted`` oversamples rare classes."""
    sampler = None
    if weighted:
        labels = dataset.labels
        class_weights = 1.0 / np.bincount(labels)
        sampler = WeightedRandomSampler(
            weights=class_weights[labels],
            num_samples=len(labels),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def to_sparse(batch, device):
    return ME.SparseTensor(batch["feats"], batch["coords"], device=device)
