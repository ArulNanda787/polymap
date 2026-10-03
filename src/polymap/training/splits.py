"""Chronological splitting. Never shuffle a temporal graph."""
from __future__ import annotations

import logging
from dataclasses import dataclass

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

    @property
    def n_cold_start(self) -> int:
        return int((~self.seen_mask).sum())

    def summary(self) -> dict:
        isolated = int((self.train_degree == 0).sum())
        return {
            "train_events": len(self.train_idx),
            "test_events": len(self.test_idx),
            "cutoff_ts": self.cutoff_ts,
            "seen_events": int(self.seen_mask.sum()),
            "cold_start_events": self.n_cold_start,
            "train_isolated_wallets": isolated,
        }


def temporal_split(bundle: GraphBundle, train_fraction: float = 0.7) -> TemporalSplit:
    ts = bundle.timestamps
    cutoff = int(np.quantile(ts, train_fraction))
    train_idx = np.where(ts <= cutoff)[0]
    test_idx = np.where(ts > cutoff)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"degenerate split at fraction {train_fraction}")

    src, tgt = bundle.edge_index
    degree = (np.bincount(src[train_idx], minlength=bundle.num_nodes)
              + np.bincount(tgt[train_idx], minlength=bundle.num_nodes))
    seen = (degree[src[test_idx]] > 0) & (degree[tgt[test_idx]] > 0)

    split = TemporalSplit(train_idx, test_idx, cutoff, seen, degree)
    log.info("split: %s", split.summary())
    return split
