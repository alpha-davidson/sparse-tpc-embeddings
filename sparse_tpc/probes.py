"""Frozen-encoder embeddings and the linear / RBF SVM probes (Section III.3)."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


@torch.no_grad()
def extract(encoder, events, task, scaling, batch_size=128):
    """512-dimensional embeddings and task labels for a list of events."""
    encoder.net.eval()
    embeddings, labels = [], []
    for start in range(0, len(events), batch_size):
        batch = events[start:start + batch_size]
        coords, feats = zip(*(encoder.featurize(e.points, scaling, task.detector) for e in batch))
        embeddings.append(encoder.embed(encoder.to_input(coords, feats)).cpu().numpy())
        labels.extend(task.label(e.raw_label) for e in batch)
    return np.concatenate(embeddings), np.array(labels)


def _scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def probe(X_train, y_train, X_test, y_test):
    """Fit both probes on standardized embeddings; score them and the naive baseline."""
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    linear = LinearSVC(C=1.0, dual=False, max_iter=100000).fit(X_train, y_train)
    rbf = SVC(C=1.0, kernel="rbf", gamma="scale").fit(X_train, y_train)
    majority = np.bincount(y_train).argmax()
    return {
        "linear": _scores(y_test, linear.predict(X_test)),
        "rbf": _scores(y_test, rbf.predict(X_test)),
        "naive": _scores(y_test, np.full_like(y_test, majority)),
    }
