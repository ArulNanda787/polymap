"""Negative samplers.

Which sampler you use changes the reported numbers more than most modelling
choices do, so they live together here with their trade-offs written down.

`random` is the conventional protocol and it is easy: a uniformly drawn wallet
is usually inactive at the relevant moment, so telling it from the true target
needs little more than knowing who is active. Treat scores under it as a
ceiling, not a result.

`hard` draws the negative from wallets the source has interacted with before.
This is the meaningful protocol, with one consequence that must be reported:
the negative is selected *using interaction history*, so any baseline scoring
on historical contact is evaluated on a set built adversarially against it and
can land below chance. That is a property of the protocol, not proof the
baseline is worthless.

`popularity` draws proportional to global degree. It is the sampler a
frequency baseline is not disadvantaged by, which is exactly why a fair
comparison needs it alongside `hard`.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class NegativeSampler:
    name = "base"

    def sample(self, source: int, rng: np.random.Generator) -> int:
        raise NotImplementedError

    def sample_many(self, sources: np.ndarray, rng: np.random.Generator,
                    n: int = 1) -> np.ndarray:
        """(len(sources), n) negatives. Subclasses override with vector code."""
        out = np.empty((len(sources), n), dtype=np.int64)
        for k, s in enumerate(sources):
            for j in range(n):
                out[k, j] = self.sample(int(s), rng)
        return out


class RandomNegativeSampler(NegativeSampler):
    name = "random"

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    def sample(self, source: int, rng) -> int:
        return int(rng.integers(0, self.num_nodes))

    def sample_many(self, sources, rng, n=1):
        return rng.integers(0, self.num_nodes, size=(len(sources), n), dtype=np.int64)


class HardNegativeSampler(NegativeSampler):
    name = "hard"

    def __init__(self, edge_index: np.ndarray, train_idx: np.ndarray, num_nodes: int):
        self.num_nodes = num_nodes
        self.history: dict[int, list[int]] = defaultdict(list)
        src, tgt = edge_index[0][train_idx], edge_index[1][train_idx]
        for s, t in zip(src, tgt):
            self.history[int(s)].append(int(t))
            self.history[int(t)].append(int(s))
        # CSR copy of the same history for vectorised sampling.
        a = np.concatenate([src, tgt]).astype(np.int64)
        b = np.concatenate([tgt, src]).astype(np.int64)
        order = np.argsort(a, kind="stable")
        self._flat = b[order]
        self._count = np.bincount(a, minlength=num_nodes).astype(np.int64)
        self._offset = np.concatenate([[0], np.cumsum(self._count)[:-1]])

    def sample(self, source: int, rng) -> int:
        partners = self.history.get(int(source))
        if not partners:      # cold-start source: fall back rather than fail
            return int(rng.integers(0, self.num_nodes))
        return int(partners[rng.integers(0, len(partners))])

    def sample_many(self, sources, rng, n=1):
        sources = np.asarray(sources, dtype=np.int64)
        cnt = self._count[sources][:, None]
        r = (rng.random((len(sources), n)) * np.maximum(cnt, 1)).astype(np.int64)
        idx = self._offset[sources][:, None] + r
        out = self._flat[np.minimum(idx, max(len(self._flat) - 1, 0))] if len(self._flat) else idx
        cold = np.broadcast_to(cnt == 0, out.shape)
        if cold.any():                      # cold-start sources: uniform fallback
            out = np.where(cold, rng.integers(0, self.num_nodes, size=out.shape), out)
        return out.astype(np.int64)


class PopularityNegativeSampler(NegativeSampler):
    name = "popularity"

    def __init__(self, edge_index: np.ndarray, train_idx: np.ndarray, num_nodes: int):
        deg = (np.bincount(edge_index[0][train_idx], minlength=num_nodes)
               + np.bincount(edge_index[1][train_idx], minlength=num_nodes))
        self.p = (deg + 1) / (deg + 1).sum()
        self._cdf = np.cumsum(self.p)
        self.num_nodes = num_nodes

    def sample(self, source: int, rng) -> int:
        return int(np.searchsorted(self._cdf, rng.random(), side="right").clip(0, self.num_nodes - 1))

    def sample_many(self, sources, rng, n=1):
        u = rng.random((len(sources), n))
        return np.searchsorted(self._cdf, u, side="right").clip(0, self.num_nodes - 1).astype(np.int64)


def build_sampler(kind: str, edge_index, train_idx, num_nodes) -> NegativeSampler:
    if kind == "random":
        return RandomNegativeSampler(num_nodes)
    if kind == "hard":
        return HardNegativeSampler(edge_index, train_idx, num_nodes)
    if kind == "popularity":
        return PopularityNegativeSampler(edge_index, train_idx, num_nodes)
    raise ValueError(f"unknown sampler '{kind}' (random | hard | popularity)")
