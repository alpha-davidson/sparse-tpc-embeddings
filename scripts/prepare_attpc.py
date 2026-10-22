"""Convert labeled AT-TPC 16O + alpha events to the GADGET II event format.

Input:
  * a zero-padded hit array of shape (events, hits, 6) whose columns 0, 1, 2
    and 4 are x, y, z and charge in detector units, and
  * a CSV with columns ``Event number`` and ``Number of tracks``.

Each labeled event keeps its nonzero hits; x and y are scaled by 35/255,
z by 740/1000, and charge by 3700/4035, so that the AT-TPC events span about
the same coordinate and charge ranges as the GADGET II events the model was
trained on. The output is an object array of (points, number_of_tracks,
event_number).

Example:
    python scripts/prepare_attpc.py --events O16_w_event_keys.npy --labels O16_labels.csv --out attpc_o16.npy
"""

import argparse

import numpy as np
import pandas as pd

SCALE = np.array([35 / 255, 35 / 255, 740 / 1000, 3700 / 4035])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", required=True, help="zero-padded AT-TPC hit array (.npy)")
    p.add_argument("--labels", required=True, help="CSV with event numbers and track counts")
    p.add_argument("--out", required=True, help="output .npy")
    args = p.parse_args()

    hits = np.load(args.events, mmap_mode="r", allow_pickle=True)
    labels = pd.read_csv(args.labels).dropna(subset=["Event number", "Number of tracks"])

    events = []
    for event_number, n_tracks in zip(labels["Event number"].astype(int), labels["Number of tracks"].astype(int)):
        event = np.asarray(hits[event_number])
        event = event[event.any(axis=1)]
        events.append((event[:, [0, 1, 2, 4]] * SCALE, n_tracks, event_number))

    out = np.empty((len(events), 3), dtype=object)
    for i, row in enumerate(events):
        out[i, 0], out[i, 1], out[i, 2] = row
    np.save(args.out, out, allow_pickle=True)
    counts = np.bincount([e[1] for e in events])
    print(f"saved {len(events)} events to {args.out}; events per track count: {counts.tolist()}")


if __name__ == "__main__":
    main()
