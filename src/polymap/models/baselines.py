"""Non-learned baselines.

A model is only as interesting as what it beats. `PopularityBaseline` is the
one most often missing, and it is the one that makes a hard-negative comparison
fair: frequency-style baselines are disadvantaged by construction under hard
negatives, popularity is not.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class Baseline:
    name = "base"

    def fit(self, edge_index, timestamps, train_idx): return self
    def score(self, source: int, target: int, at_time: int) -> float:
        raise NotImplementedError

    def score_many(self, sources, targets, times) -> np.ndarray:
        return np.array([self.score(s, t, ts)
                         for s, t, ts in zip(sources, targets, times)], dtype=float)


class HistoricalFrequencyBaseline(Baseline):
    name = "historical_frequency"

    def fit(self, edge_index, timestamps, train_idx):
        self.counts = defaultdict(int)
        for s, t in zip(edge_index[0][train_idx], edge_index[1][train_idx]):
            self.counts[(int(s), int(t))] += 1
            self.counts[(int(t), int(s))] += 1
        return self

    def score(self, source, target, at_time=None):
        return float(self.counts.get((int(source), int(target)), 0))


class RecencyFrequencyBaseline(Baseline):
    name = "recency_frequency"

    def __init__(self, half_life_hours: float = 24.0):
        self.half_life = half_life_hours * 3600.0

    def fit(self, edge_index, timestamps, train_idx):
        self.events = defaultdict(list)
        for i in train_idx:
            s, t, ts = int(edge_index[0][i]), int(edge_index[1][i]), int(timestamps[i])
            self.events[(s, t)].append(ts)
            self.events[(t, s)].append(ts)
        return self

    def score(self, source, target, at_time):
        times = self.events.get((int(source), int(target)))
        if not times:
            return 0.0
        dt = np.maximum(np.array(at_time) - np.array(times), 0)
        return float(np.sum(0.5 ** (dt / self.half_life)))


class PopularityBaseline(Baseline):
    """Ranks purely on how connected the target is. Ignores the source entirely.

    This is the baseline the hard-negative protocol does *not* target, so it is
    the honest reference point for the model's advantage.
    """
    name = "popularity"

    def fit(self, edge_index, timestamps, train_idx):
        n = int(max(edge_index.max(), 0)) + 1
        self.degree = (np.bincount(edge_index[0][train_idx], minlength=n)
                       + np.bincount(edge_index[1][train_idx], minlength=n))
        return self

    def score(self, source, target, at_time=None):
        t = int(target)
        return float(self.degree[t]) if t < len(self.degree) else 0.0


ALL_BASELINES = {
    "historical_frequency": HistoricalFrequencyBaseline,
    "recency_frequency": RecencyFrequencyBaseline,
    "popularity": PopularityBaseline,
}
