"""Non-learned baselines for temporal link prediction.

A model is only as interesting as what it beats. Four scorers, each answering
"how much would a simple rule have said about this pair?":

  pair_count      how many times these two wallets co-traded before
  recency         the same, but old co-trades decay (half-life, default 24h)
  popularity      how connected the candidate is; ignores the source entirely
  common_neighbours  how many wallets both have co-traded with

Protocol (matches the TGNN in `trainer.score_events`)
-----------------------------------------------------
`fit` ingests the TRAIN events. During evaluation, events are scored in
chronological batches and a batch is `update`d into the state only AFTER it has
been scored. The TGNN absorbs test events into memory the same way, so a
baseline fit once on train and frozen would be handicapped. All state is
symmetric (undirected), because the graph's source/target order carries no
meaning.

`score_candidates(sources, candidates, at_time)` takes (n,) sources and an
(n, k) matrix of candidate targets and returns (n, k) scores, so positives and
negatives go through exactly the same code path.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class Baseline:
    name = "base"
    label = "base"

    def fit(self, edge_index, timestamps, train_idx):
        """Ingest the train events, in index (= chronological) order."""
        self._n = int(edge_index.max()) + 1 if edge_index.size else 0
        self._init_state()
        self.update(edge_index[0][train_idx], edge_index[1][train_idx],
                    timestamps[train_idx])
        return self

    # --- subclass API ---------------------------------------------------
    def _init_state(self): ...
    def update(self, sources, targets, times): raise NotImplementedError
    def score(self, source: int, target: int, at_time: int = 0) -> float:
        raise NotImplementedError

    # --- shared ----------------------------------------------------------
    def score_candidates(self, sources, candidates, at_time=None) -> np.ndarray:
        sources = np.asarray(sources)
        candidates = np.asarray(candidates)
        if candidates.ndim == 1:
            candidates = candidates[:, None]
        times = np.zeros(len(sources), dtype=np.int64) if at_time is None else np.asarray(at_time)
        out = np.empty(candidates.shape, dtype=np.float64)
        for i in range(len(sources)):
            s, ts = int(sources[i]), int(times[i])
            row = candidates[i]
            for j in range(row.shape[0]):
                out[i, j] = self.score(s, int(row[j]), ts)
        return out

    def score_many(self, sources, targets, times) -> np.ndarray:
        return np.array([self.score(int(s), int(t), int(ts))
                         for s, t, ts in zip(sources, targets, times)], dtype=float)


def _key(a: int, b: int, n: int) -> int:
    return (a * n + b) if a < b else (b * n + a)


class PairCountBaseline(Baseline):
    name = "pair_count"
    label = "Pair count"

    def _init_state(self):
        self.counts: dict[int, int] = defaultdict(int)

    def update(self, sources, targets, times):
        n = self._n
        for s, t in zip(np.asarray(sources).tolist(), np.asarray(targets).tolist()):
            self.counts[_key(s, t, n)] += 1

    def score(self, source, target, at_time=0):
        return float(self.counts.get(_key(int(source), int(target), self._n), 0))


class RecencyBaseline(Baseline):
    """Exponentially decayed count: sum over past co-trades of 0.5 ** (age / half_life).

    Kept incrementally as (decayed_sum, last_time), which is exactly equal to
    the full sum because decay composes.
    """
    name = "recency"
    label = "Recency-weighted count"

    def __init__(self, half_life_hours: float = 24.0):
        self.half_life = half_life_hours * 3600.0

    def _init_state(self):
        self.state: dict[int, tuple[float, int]] = {}

    def update(self, sources, targets, times):
        n, hl = self._n, self.half_life
        for s, t, ts in zip(np.asarray(sources).tolist(), np.asarray(targets).tolist(),
                            np.asarray(times).tolist()):
            k = _key(s, t, n)
            prev = self.state.get(k)
            if prev is None:
                self.state[k] = (1.0, ts)
            else:
                acc, last = prev
                self.state[k] = (acc * 0.5 ** (max(ts - last, 0) / hl) + 1.0, max(ts, last))

    def score(self, source, target, at_time=0):
        prev = self.state.get(_key(int(source), int(target), self._n))
        if prev is None:
            return 0.0
        acc, last = prev
        return float(acc * 0.5 ** (max(int(at_time) - last, 0) / self.half_life))


class PopularityBaseline(Baseline):
    """Ranks purely on how connected the candidate is. Ignores the source.

    The one baseline the hard-negative protocol does not target, so it is the
    honest reference point for the model's advantage.
    """
    name = "popularity"
    label = "Popularity (degree)"

    def _init_state(self):
        self.degree = np.zeros(self._n, dtype=np.int64)

    def update(self, sources, targets, times):
        np.add.at(self.degree, np.asarray(sources, dtype=np.int64), 1)
        np.add.at(self.degree, np.asarray(targets, dtype=np.int64), 1)

    def score(self, source, target, at_time=0):
        t = int(target)
        return float(self.degree[t]) if 0 <= t < len(self.degree) else 0.0

    def score_candidates(self, sources, candidates, at_time=None):
        c = np.asarray(candidates)
        if c.ndim == 1:
            c = c[:, None]
        return self.degree[c].astype(np.float64)


class CommonNeighboursBaseline(Baseline):
    """Number of wallets both endpoints have co-traded with (classic link-prediction score)."""
    name = "common_neighbours"
    label = "Common neighbours"

    def _init_state(self):
        self.nbrs: dict[int, set[int]] = defaultdict(set)

    def update(self, sources, targets, times):
        for s, t in zip(np.asarray(sources).tolist(), np.asarray(targets).tolist()):
            if s != t:
                self.nbrs[s].add(t)
                self.nbrs[t].add(s)

    def score(self, source, target, at_time=0):
        a = self.nbrs.get(int(source))
        b = self.nbrs.get(int(target))
        if not a or not b:
            return 0.0
        return float(len(a & b))


# Public registry, in the order they appear in tables.
ALL_BASELINES = {
    "pair_count": PairCountBaseline,
    "recency": RecencyBaseline,
    "popularity": PopularityBaseline,
    "common_neighbours": CommonNeighboursBaseline,
}

# Old names, kept so existing notebooks that import them still work.
HistoricalFrequencyBaseline = PairCountBaseline
RecencyFrequencyBaseline = RecencyBaseline
