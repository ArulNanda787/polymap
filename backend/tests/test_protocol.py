"""Tests for the evaluation protocol. Numpy/pandas only (no torch)."""
import numpy as np
import pandas as pd
import pytest

from polymap.evaluation.protocol import (block_bootstrap, build_eval_negatives, dedupe_mask,
                                         time_grouped_slices)
from polymap.graph.build import GraphBundle
from polymap.models import baselines_extra  # noqa: F401
from polymap.models.baselines import ALL_BASELINES
from polymap.training.splits import TemporalSplit


def _bundle(n_nodes=30, n_edges=400, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(1000, 1000 + 60, n_edges))          # many simultaneous timestamps
    s = rng.integers(0, n_nodes, n_edges)
    t = (s + rng.integers(1, n_nodes, n_edges)) % n_nodes          # never a self loop
    return GraphBundle(x=np.zeros((n_nodes, 2), np.float32),
                       edge_index=np.vstack([s, t]).astype(np.int64),
                       edge_attr=np.zeros((n_edges, 4), np.float32), timestamps=ts.astype(np.int64),
                       wallet_to_id={}, id_to_wallet={}, edges=pd.DataFrame(), scaler=None)


def _split(b, frac=0.6):
    ts = b.timestamps
    cut = int(np.quantile(np.unique(ts), frac))
    tr, te = np.where(ts <= cut)[0], np.where(ts > cut)[0]
    deg = np.bincount(b.edge_index[0][tr], minlength=b.num_nodes) + np.bincount(
        b.edge_index[1][tr], minlength=b.num_nodes)
    return TemporalSplit(tr, te, cut, np.ones(len(te), bool), deg)


@pytest.mark.parametrize("kind", ["random", "popularity", "hard", "active", "inductive"])
def test_negatives_never_collide_with_same_timestamp_partners(kind):
    b = _bundle()
    sp = _split(b)
    ns = build_eval_negatives(b, sp, kind, n_neg=10, seed=1)
    src, tgt, ts = b.edge_index[0], b.edge_index[1], b.timestamps
    for row, i in enumerate(sp.test_idx):
        same = (ts == ts[i]) & ((src == src[i]) | (tgt == src[i]))
        partners = set(tgt[same & (src == src[i])]) | set(src[same & (tgt == src[i])])
        assert src[i] not in ns.ids[row]
        assert not (partners & set(ns.ids[row].tolist()))


def test_negatives_are_reproducible_and_independent_of_nothing_else():
    b = _bundle()
    sp = _split(b)
    a1 = build_eval_negatives(b, sp, "hard", 8, seed=5).ids
    a2 = build_eval_negatives(b, sp, "hard", 8, seed=5).ids
    assert np.array_equal(a1, a2)


def test_time_grouped_slices_never_split_a_timestamp():
    ts = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4])
    for max_events in (1, 2, 4, 100):
        sl = time_grouped_slices(ts, max_events)
        assert sum(s.stop - s.start for s in sl) == len(ts)
        for s in sl[:-1]:
            assert ts[s.stop - 1] != ts[s.stop]


def test_dedupe_keeps_one_row_per_pair_and_timestamp():
    b = _bundle()
    b.edge_index[:, :] = b.edge_index                              # noqa
    sp = _split(b)
    m = dedupe_mask(b, sp)
    s, t = b.edge_index[0][sp.test_idx], b.edge_index[1][sp.test_idx]
    keys = set(zip(np.minimum(s, t)[m], np.maximum(s, t)[m], b.timestamps[sp.test_idx][m]))
    assert len(keys) == m.sum()


def test_edgebank_is_causal_and_binary():
    ei = np.array([[0, 1], [1, 2]])
    ts = np.array([10, 20])
    eb = ALL_BASELINES["edgebank_unlimited"]().fit(ei, ts, np.array([0]))
    assert eb.score(0, 1) == 1.0 and eb.score(1, 2) == 0.0      # edge 1 not yet absorbed
    eb.update([1], [2], [20])
    assert eb.score(1, 2) == 1.0
    w = ALL_BASELINES["edgebank_1h"]().fit(ei, ts, np.array([0]))
    assert w.score(0, 1, at_time=10 + 3000) == 1.0
    assert w.score(0, 1, at_time=10 + 5000) == 0.0


def test_block_bootstrap_brackets_the_mean():
    d = np.random.default_rng(0).normal(0.05, 0.2, 2000)
    r = block_bootstrap(d, n_boot=500)
    assert r["ci95_low"] < r["mean"] < r["ci95_high"]
