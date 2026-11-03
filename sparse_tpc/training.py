"""Train an encoder on its source task (Appendix A, Table 6 of the paper)."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass
class TrainConfig:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 200
    scheduler_factor: float = 0.1
    scheduler_patience: int = 10
    scheduler_threshold: float = 1e-3
    min_lr: float = 5e-8
    early_stopping_patience: int = 30
    grad_clip: float = 1.0  # max gradient norm; 0 disables clipping
    weighted_sampler: bool = True
    drop_last: bool = False  # drop the last incomplete training batch
    num_workers: int = 1


class _EventDataset(Dataset):
    def __init__(self, encoder, events, task, scaling, augment):
        self.encoder, self.events, self.task = encoder, events, task
        self.scaling, self.augment = scaling, augment

    def __len__(self):
        return len(self.events)

    def __getitem__(self, i):
        e = self.events[i]
        coords, feats = self.encoder.featurize(e.points, self.scaling, self.task.detector, self.augment)
        return coords, feats, self.task.label(e.raw_label)


def _collate(batch):
    return list(zip(*batch))


def _seed_worker(worker_id):
    # Fresh NumPy stream per worker and epoch for the random translations.
    np.random.seed(torch.initial_seed() % 2**32)


def make_loader(encoder, events, task, scaling, cfg, train=False):
    dataset = _EventDataset(encoder, events, task, scaling, augment=train)
    sampler = None
    if train and cfg.weighted_sampler:
        labels = np.array([task.label(e.raw_label) for e in events])
        weights = (1.0 / np.bincount(labels))[labels]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights))
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
        worker_init_fn=_seed_worker,
        drop_last=train and cfg.drop_last,
    )


def _run_batches(encoder, loader, criterion, optimizer=None, grad_clip=0.0):
    total_loss, y_true, y_pred = 0.0, [], []
    for coords, feats, labels in loader:
        x = encoder.to_input(coords, feats)
        target = torch.as_tensor(labels, device=encoder.device)
        if optimizer is not None:
            optimizer.zero_grad()
        logits = encoder.logits(x)
        loss = criterion(logits, target)
        if optimizer is not None:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(encoder.net.parameters(), max_norm=grad_clip)
            optimizer.step()
        total_loss += loss.item()
        y_true.extend(labels)
        y_pred.extend(logits.argmax(1).tolist())
    return total_loss / max(len(loader), 1), np.array(y_true), np.array(y_pred)


def train_encoder(encoder, task, train_events, val_events, scaling, cfg, checkpoint_path, log=print):
    """Train with early stopping and keep the lowest-validation-loss checkpoint."""
    net = encoder.net
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        threshold=cfg.scheduler_threshold,
        threshold_mode="rel",
        min_lr=cfg.min_lr,
    )
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = make_loader(encoder, train_events, task, scaling, cfg, train=True)
    val_loader = make_loader(encoder, val_events, task, scaling, cfg)

    best_loss, best_epoch, stale, history = float("inf"), -1, 0, []
    for epoch in range(cfg.max_epochs):
        net.train()
        train_loss, _, _ = _run_batches(encoder, train_loader, criterion, optimizer, cfg.grad_clip)
        net.eval()
        with torch.no_grad():
            val_loss, y_true, y_pred = _run_batches(encoder, val_loader, criterion)
        val_acc = float(np.mean(y_true == y_pred))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc})
        log(f"epoch {epoch:3d}  train loss {train_loss:.5f}  val loss {val_loss:.5f}  "
            f"val acc {val_acc:.4f}  lr {optimizer.param_groups[0]['lr']:.1e}")

        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save(net.state_dict(), checkpoint_path)
        else:
            stale += 1
        scheduler.step(val_loss)
        if cfg.early_stopping_patience > 0 and stale >= cfg.early_stopping_patience:
            log(f"early stopping at epoch {epoch}")
            break

    net.load_state_dict(torch.load(checkpoint_path, map_location=encoder.device))
    return {"best_epoch": best_epoch, "best_val_loss": best_loss,
            "epochs_ran": len(history), "history": history}


@torch.no_grad()
def evaluate(encoder, task, events, scaling, cfg):
    """Accuracy of the source-task classifier on held-out events."""
    encoder.net.eval()
    loader = make_loader(encoder, events, task, scaling, cfg)
    _, y_true, y_pred = _run_batches(encoder, loader, torch.nn.CrossEntropyLoss())
    return float(np.mean(y_true == y_pred))
