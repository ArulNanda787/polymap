"""Chronological splitting. Never shuffle a temporal graph."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np

from ..graph.build import GraphBundle

log = logging.getLogger(__name__)


@dataclass
class TemporalSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    cutoff_ts: int
    seen_mask: np.ndarray      # test events whose wallets both appear in train
    train_degree: np.ndarray
    validation_idx: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def n_cold_start(self) -> int:
        return int((~self.seen_mask).sum())

    def summary(self) -> dict:
        isolated = int((self.train_degree == 0).sum())
        return {
            "train_events": len(self.train_idx),
            "test_events": len(self.test_idx),
            "validation_events": len(self.validation_idx),
            "cutoff_ts": self.cutoff_ts,
            "seen_events": int(self.seen_mask.sum()),
            "cold_start_events": self.n_cold_start,
            "train_isolated_wallets": isolated,
        }


def temporal_split(bundle: GraphBundle, train_fraction: float = 0.7,
                   validation_fraction: float = 0.0) -> TemporalSplit:
    ts = bundle.timestamps
    uniq, counts = np.unique(ts, return_counts=True)
    if len(uniq) < 2:
        raise ValueError(f"degenerate split at fraction {train_fraction}")
    # Cut only between distinct timestamps, choosing the boundary whose train
    # share is closest to train_fraction (one burst can hold ~10% of all events).
    cum = np.cumsum(counts) / len(ts)
    k = int(np.argmin(np.abs(cum - train_fraction)))
    k = min(k, len(uniq) - 2)             # keep at least one test timestamp
    cutoff = int(uniq[k])
    train_idx = np.where(ts <= cutoff)[0]
    test_idx = np.where(ts > cutoff)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"degenerate split at fraction {train_fraction}")

    src, tgt = bundle.edge_index
    degree = (np.bincount(src[train_idx], minlength=bundle.num_nodes)
              + np.bincount(tgt[train_idx], minlength=bundle.num_nodes))
    seen = (degree[src[test_idx]] > 0) & (degree[tgt[test_idx]] > 0)

    validation_idx = np.empty(0, dtype=np.int64)
    if validation_fraction:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        train_ts = ts[train_idx]
        val_uniq, val_counts = np.unique(train_ts, return_counts=True)
        if len(val_uniq) < 2:
            raise ValueError("validation split needs at least two train timestamps")
        target = 1.0 - validation_fraction
        k_val = int(np.argmin(np.abs(np.cumsum(val_counts) / len(train_idx) - target)))
        k_val = min(k_val, len(val_uniq) - 2)
        val_cutoff = int(val_uniq[k_val])
        validation_idx = np.where((ts > val_cutoff) & (ts <= cutoff))[0]
        if len(validation_idx) == 0:
            raise ValueError("validation split produced no events; reduce validation_fraction")

    split = TemporalSplit(train_idx, test_idx, cutoff, seen, degree, validation_idx)
    log.info("split: %s", split.summary())
    return split


def validation_view(split: TemporalSplit) -> TemporalSplit:
    """Return a split that fits on the pre-validation prefix and scores validation."""
    if len(split.validation_idx) == 0:
        raise ValueError("split has no validation events")
    fit_idx = np.setdiff1d(split.train_idx, split.validation_idx, assume_unique=True)
    return replace(split, train_idx=fit_idx, test_idx=split.validation_idx,
                   validation_idx=np.empty(0, dtype=np.int64))
