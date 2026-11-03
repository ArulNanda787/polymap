"""Causal pair-history features used by the link-prediction head.

The graph memory is node-level, so two wallets that have just interacted can
still receive almost identical node representations.  This small state object
provides the missing pair-level, time-aware signal without looking ahead:
features for an event are read first, then the event is added to the history.

The six features are, in order:

    log(1 + pair count), decayed count at 5 minutes, 1 hour and 24 hours,
    seen flag, and log(1 + pair age in hours).

All state is symmetric because co-trading edges represent an undirected wallet
relationship for this model.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

PAIR_FEATURE_DIM = 6
HALF_LIVES_SECONDS = (5.0 * 60.0, 60.0 * 60.0, 24.0 * 60.0 * 60.0)


def _key(a: int, b: int, n_nodes: int) -> int:
    return (a * n_nodes + b) if a < b else (b * n_nodes + a)


class PairHistory:
    """Incremental, causal history for undirected wallet pairs."""

    def __init__(self, n_nodes: int):
        self.n_nodes = int(n_nodes)
        self.counts: dict[int, int] = defaultdict(int)
        # key -> (last event time, decayed count at each configured half-life)
        self.state: dict[int, tuple[int, np.ndarray]] = {}

    def _features_one(self, source: int, target: int, at_time: int) -> np.ndarray:
        key = _key(int(source), int(target), self.n_nodes)
        previous = self.state.get(key)
        if previous is None:
            return np.zeros(PAIR_FEATURE_DIM, dtype=np.float32)

        last, decayed = previous
        age = max(int(at_time) - last, 0)
        out = np.empty(PAIR_FEATURE_DIM, dtype=np.float32)
        out[0] = np.log1p(self.counts[key])
        out[1:4] = decayed * np.power(0.5, age / np.asarray(HALF_LIVES_SECONDS))
        out[4] = 1.0
        out[5] = np.log1p(age / 3600.0)
        return out

    def features(self, sources, targets, times) -> np.ndarray:
        """Return causal features for aligned arrays or broadcastable inputs."""
        sources, targets, times = np.broadcast_arrays(
            np.asarray(sources), np.asarray(targets), np.asarray(times))
        flat = [self._features_one(s, t, ts)
                for s, t, ts in zip(sources.flat, targets.flat, times.flat)]
        return np.asarray(flat, dtype=np.float32).reshape(sources.shape + (PAIR_FEATURE_DIM,))

    def update(self, sources, targets, times) -> None:
        """Absorb events after they have been scored."""
        for source, target, timestamp in zip(np.asarray(sources).flat,
                                             np.asarray(targets).flat,
                                             np.asarray(times).flat):
            source, target, timestamp = int(source), int(target), int(timestamp)
            key = _key(source, target, self.n_nodes)
            previous = self.state.get(key)
            if previous is None:
                decayed = np.ones(3, dtype=np.float64)
            else:
                last, old = previous
                age = max(timestamp - last, 0)
                decayed = old * np.power(0.5, age / np.asarray(HALF_LIVES_SECONDS)) + 1.0
            self.counts[key] += 1
            self.state[key] = (timestamp, decayed)

    def replay(self, edge_index: np.ndarray, timestamps: np.ndarray,
               indices: np.ndarray) -> None:
        """Absorb an indexed chronological event prefix."""
        src, tgt = edge_index
        self.update(src[indices], tgt[indices], timestamps[indices])
