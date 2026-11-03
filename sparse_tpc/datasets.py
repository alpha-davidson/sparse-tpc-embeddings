"""Datasets, labels, splits and input scaling for the GADGET II and AT-TPC tasks.

Each dataset is a folder of per-event ``<event_number>.npy`` files. Every file
holds an (N, 7) float array with one row per hit and the columns

    x, y, z, charge, event number, raw label, number of hits

with positions in mm. The raw label is the decay branch (GADGET II) or the
number of tracks (AT-TPC); each task maps it to its own classes.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

# Fixed coordinate ranges (mm) used to scale positions to [0, 1].
DETECTOR_RANGES = {
    "gadget": {"MIN_X": -38.5, "MAX_X": 34.1, "MIN_Y": -38.5, "MAX_Y": 34.1},
    "attpc": {
        "MIN_X": -270.0, "MAX_X": 270.0,
        "MIN_Y": -270.0, "MAX_Y": 270.0,
        "MIN_Z": -185.0, "MAX_Z": 1155.0,
    },
}


@dataclass(frozen=True)
class Task:
    name: str
    detector: str
    folder: str
    class_names: tuple
    raw_to_class: dict

    @property
    def num_classes(self):
        return len(self.class_names)

    def label(self, raw):
        return self.raw_to_class[int(raw)]


TASKS = {
    t.name: t
    for t in (
        Task("Mg20-2", "gadget", "Mg20-3_filtered_full_variable",
             ("alpha", "proton"), {0: 0, 1: 1, 2: 1}),
        Task("Mg20-3", "gadget", "Mg20-3_filtered_full_variable",
             ("2153 keV alpha", "770 keV p", "1596 keV p"), {0: 0, 1: 1, 2: 2}),
        Task("Mg21-3", "gadget", "Mg21-3_filtered_full_variable",
             ("2153 keV alpha", "1285 keV p", "1930 keV p"), {0: 0, 1: 1, 2: 2}),
        Task("O16-2", "attpc", "O16-3_filtered_full_variable",
             ("0-2 tracks", "3-5 tracks"), {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}),
        Task("O16-3", "attpc", "O16-3_filtered_full_variable",
             ("0-2 tracks", "3 tracks", "4-5 tracks"), {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2}),
        Task("B10-2", "attpc", "B10-2_filtered_full_variable",
             ("3 tracks", "4 tracks"), {3: 0, 4: 1}),
    )
}


def same_experiment(a, b):
    """True if two tasks are label variants of the same events (e.g. O16-2 and O16-3)."""
    return TASKS[a].folder == TASKS[b].folder


@dataclass
class Event:
    points: np.ndarray  # (N, 4): x, y, z, charge
    raw_label: int
    event_id: int


def load_folder(data_root, folder):
    """Load every event in a dataset folder, sorted by event number."""
    files = sorted(Path(data_root, folder).glob("*.npy"), key=lambda f: int(f.stem))
    if not files:
        raise FileNotFoundError(f"no .npy events in {Path(data_root, folder)}")
    events = []
    for f in files:
        arr = np.load(f)
        events.append(Event(arr[:, :4].astype(np.float32), int(arr[0, 5]), int(f.stem)))
    return events


def kfold_split(events, fold, k=5, val_frac=0.2, seed=28):
    """Fold ``fold`` of a stratified k-fold split: (train, val, test) event lists.

    The fold is the held-out test set; the other k - 1 folds are split into
    training and validation sets. ``seed`` must be the same for every fold so
    that the k test sets partition the data.
    """
    raw = np.array([e.raw_label for e in events])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    trainval_idx, test_idx = list(skf.split(np.zeros(len(events)), raw))[fold]
    train_idx, val_idx = train_test_split(
        trainval_idx, train_size=1 - val_frac, stratify=raw[trainval_idx], random_state=seed
    )
    return ([events[i] for i in train_idx], [events[i] for i in val_idx],
            [events[i] for i in test_idx])


def scaling_constants(events, detector):
    """Charge (and GADGET II drift) scaling constants from a set of events.

    Only pass training (and validation) events, so nothing about the
    evaluation events leaks into the scaling.
    """
    amps = np.concatenate([e.points[:, 3] for e in events])
    s = dict(DETECTOR_RANGES[detector])
    s["MIN_LOG_A"] = np.log(np.min(amps + 1e-10))
    s["MAX_LOG_A"] = np.log(np.max(amps + 1e-10))
    if detector == "gadget":
        # GADGET II events have no absolute drift reference: measure z from
        # each event's first hit, standardize, then map to [0, 1].
        rel_z = [e.points[:, 2] - e.points[:, 2].min() for e in events]
        all_z = np.concatenate(rel_z)
        s["MEAN_Z"], s["STD_Z"] = np.mean(all_z), np.std(all_z)
        z_std = np.concatenate([(z - s["MEAN_Z"]) / s["STD_Z"] for z in rel_z])
        s["Z_SCALED_MIN"], s["Z_SCALED_MAX"] = np.min(z_std), np.max(z_std)
    return s


def scale_points(xyz, charge, s, detector):
    """Scale positions to about [0, 1] and log charge to [0, 1]; returns (N, 4)."""
    out = np.empty((len(xyz), 4), dtype=np.float32)
    out[:, 0] = (xyz[:, 0] - s["MIN_X"]) / (s["MAX_X"] - s["MIN_X"])
    out[:, 1] = (xyz[:, 1] - s["MIN_Y"]) / (s["MAX_Y"] - s["MIN_Y"])
    if detector == "gadget":
        z_std = ((xyz[:, 2] - xyz[:, 2].min()) - s["MEAN_Z"]) / s["STD_Z"]
        out[:, 2] = (z_std - s["Z_SCALED_MIN"]) / (s["Z_SCALED_MAX"] - s["Z_SCALED_MIN"])
    else:
        out[:, 2] = (xyz[:, 2] - s["MIN_Z"]) / (s["MAX_Z"] - s["MIN_Z"])
    out[:, 3] = (np.log(charge + 1e-10) - s["MIN_LOG_A"]) / (s["MAX_LOG_A"] - s["MIN_LOG_A"])
    return out
