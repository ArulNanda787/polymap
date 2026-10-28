"""Baselines, tie-aware ranking, and the shared-negatives comparison."""
from types import SimpleNamespace

import numpy as np
import pytest

from polymap.evaluation.baseline_eval import evaluate_scores, paired_mrr_gap, score_baselines
from polymap.evaluation.metrics import per_event_ranking, ranking_metrics
from polymap.models.baselines import ALL_BASELINES


# ----------------------------------------------------------- ranking ties --
def test_ranking_without_ties_is_the_usual_rank():
    pos = np.array([0.9, 0.2])
    neg = np.array([[0.1, 0.5, 0.3], [0.1, 0.5, 0.3]])
    m = ranking_metrics(pos, neg)
    assert m["mrr"] == pytest.approx((1 / 1 + 1 / 3) / 2)
    assert m["hits@1"] == pytest.approx(0.5)


def test_all_tied_scores_are_not_a_free_win():
    # Positive ties with 4 negatives: rank uniform on 1..5.
    ev = per_event_ranking(np.array([0.0]), np.zeros((1, 4)))
    assert ev["rr"][0] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5) / 5)
    assert ev["hits@1"][0] == pytest.approx(1 / 5)
    assert ev["hits@5"][0] == pytest.approx(1.0)
    assert ev["mean_rank"][0] == pytest.approx(3.0)


def test_partial_ties():
    # 1 negative strictly above, 2 tied: rank uniform on 2..4
    ev = per_event_ranking(np.array([0.5]), np.array([[0.9, 0.5, 0.5, 0.1]]))
    assert ev["rr"][0] == pytest.approx((1 / 2 + 1 / 3 + 1 / 4) / 3)
    assert ev["hits@1"][0] == 0.0
    assert ev["hits@5"][0] == 1.0
    ev2 = per_event_ranking(np.array([0.5]), np.array([[0.9, 0.5, 0.5, 0.1]]), ks=(2, 3))
    assert ev2["hits@2"][0] == pytest.approx(1 / 3)     # only rank 2 of {2,3,4} fits
    assert ev2["hits@3"][0] == pytest.approx(2 / 3)


# --------------------------------------------------------------- baselines --
def _fit(cls, src, tgt, ts, n_train):
    ei = np.array([src, tgt], dtype=np.int64)
    return cls().fit(ei, np.array(ts, dtype=np.int64), np.arange(n_train)), ei


def test_pair_count_is_symmetric_and_counts_train_only():
    m, _ = _fit(ALL_BASELINES["pair_count"], [0, 1, 0, 2], [1, 0, 2, 3], [1, 2, 3, 4], 3)
    assert m.score(0, 1) == m.score(1, 0) == 2.0
    assert m.score(0, 2) == 1.0
    assert m.score(2, 3) == 0.0           # event index 3 is not in train


def test_recency_matches_brute_force_sum():
    ts = [0, 3600, 7200, 90_000]
    m, _ = _fit(ALL_BASELINES["recency"], [0, 1, 0, 1], [1, 0, 1, 0], ts, 4)
    now = 100_000
    hl = 24 * 3600
    brute = sum(0.5 ** ((now - t) / hl) for t in ts)
    assert m.score(0, 1, now) == pytest.approx(brute)
    assert m.score(0, 2, now) == 0.0


def test_popularity_ignores_source_and_updates_online():
    m, _ = _fit(ALL_BASELINES["popularity"], [0, 0, 1], [1, 2, 2], [1, 2, 3], 3)
    assert m.score(5, 2) == m.score(0, 2) == 2.0
    m.update(np.array([0]), np.array([2]), np.array([9]))
    assert m.score(0, 2) == 3.0


def test_common_neighbours():
    # 0-1, 1-2, 0-3, 3-2  => 0 and 2 share neighbours {1, 3}
    m, _ = _fit(ALL_BASELINES["common_neighbours"], [0, 1, 0, 3], [1, 2, 3, 2], [1, 2, 3, 4], 4)
    assert m.score(0, 2) == 2.0
    assert m.score(0, 1) == 0.0
    assert m.score(0, 9) == 0.0


# --------------------------------------------------- shared-negatives eval --
def _toy(n_nodes=30, n_edges=400, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n_nodes, n_edges)
    tgt = (src + rng.integers(1, n_nodes, n_edges)) % n_nodes
    ts = np.sort(rng.integers(0, 10_000_000, n_edges))
    bundle = SimpleNamespace(edge_index=np.vstack([src, tgt]).astype(np.int64),
                             timestamps=ts.astype(np.int64), num_nodes=n_nodes)
    cut = int(0.7 * n_edges)
    split = SimpleNamespace(train_idx=np.arange(cut), test_idx=np.arange(cut, n_edges))
    neg = rng.integers(0, n_nodes, (n_edges - cut, 7)).astype(np.int64)
    return bundle, split, neg


def test_score_baselines_shapes_and_no_leakage():
    bundle, split, neg = _toy()
    out = score_baselines(bundle, split, neg, batch_size=4)
    assert set(out) == set(ALL_BASELINES)
    n = len(split.test_idx)
    for v in out.values():
        assert v["positive"].shape == (n,) and v["negative"].shape == (n, 7)

    # Leakage check: with batch_size=1, a test event's own pair count must not
    # include itself. Recompute pair_count by brute force from strictly earlier events.
    out1 = score_baselines(bundle, split, neg, batch_size=1, names=("pair_count",))
    src, tgt = bundle.edge_index
    for j, i in enumerate(split.test_idx[:50]):
        a, b = sorted((src[i], tgt[i]))
        earlier = sum(1 for k in range(i) if sorted((src[k], tgt[k])) == [a, b])
        assert out1["pair_count"]["positive"][j] == earlier


def test_score_baselines_rejects_misaligned_negatives():
    bundle, split, neg = _toy()
    with pytest.raises(ValueError, match="test events"):
        score_baselines(bundle, split, neg[:-1], batch_size=4)
    with pytest.raises(ValueError, match="unknown baseline"):
        score_baselines(bundle, split, neg, batch_size=4, names=("nope",))


def test_evaluate_scores_has_expected_keys():
    rng = np.random.default_rng(1)
    pos, neg = rng.normal(1, 1, 200), rng.normal(0, 1, (200, 10))
    gap = np.where(rng.random(200) < 0.3, np.nan, rng.random(200) * 1000)
    m = evaluate_scores(pos, neg, gap, window_s=300)
    for k in ("roc_auc", "average_precision", "mrr", "hits@1", "hits@10", "by_recency"):
        assert k in m
    assert m["roc_auc"] > 0.5
    assert sum(v["n"] for v in m["by_recency"].values()) == 200


def test_paired_gap_detects_a_real_difference_and_a_null():
    rng = np.random.default_rng(2)
    n = 1500
    neg = rng.normal(0, 1, (n, 20))
    good = rng.normal(2.0, 1, n)
    bad = rng.normal(0.0, 1, n)
    big = paired_mrr_gap(good, neg, bad, neg, n_boot=400)
    assert big["ci95_low"] > 0
    # identical scorers -> gap exactly 0, interval contains 0
    same = paired_mrr_gap(good, neg, good, neg, n_boot=400)
    assert same["mrr_gap"] == 0.0 and same["ci95_low"] <= 0 <= same["ci95_high"]


# ----------------------------------------------- TGNN + baselines together --
def test_tgnn_and_baselines_share_the_same_negatives():
    pytest.importorskip("torch")
    from polymap.config import ModelConfig
    from polymap.features.edges import build_temporal_edges
    from polymap.features.wallets import build_wallet_features
    from polymap.graph.build import build_graph
    from polymap.training.negatives import build_sampler
    from polymap.training.splits import temporal_split
    from polymap.training.trainer import score_events, train
    from tests.test_edges import make_trades

    e = build_temporal_edges(make_trades(700, 4), window_seconds=300)
    b = build_graph(e, build_wallet_features(make_trades(700, 4)))
    s = temporal_split(b, 0.7)
    r = train(b, s, ModelConfig(epochs=1, batch_size=64, device="cpu"), "random")
    sampler = build_sampler("hard", b.edge_index, s.train_idx, b.num_nodes)
    sc = score_events(r, b, s, sampler, n_negatives=6, batch_size=4)
    assert sc["neg_ids"].shape == (len(s.test_idx), 6)
    # no negative equals its own source or true target
    src, tgt = b.edge_index[0][s.test_idx], b.edge_index[1][s.test_idx]
    assert not (sc["neg_ids"] == src[:, None]).any()
    assert not (sc["neg_ids"] == tgt[:, None]).any()
    base = score_baselines(b, s, sc["neg_ids"], batch_size=4)
    for v in base.values():
        assert v["negative"].shape == sc["negative"].shape


def test_pair_history_head_scores_causally():
    pytest.importorskip("torch")
    from polymap.config import ModelConfig
    from polymap.features.edges import build_temporal_edges
    from polymap.features.wallets import build_wallet_features
    from polymap.graph.build import build_graph
    from polymap.training.negatives import build_sampler
    from polymap.training.splits import temporal_split
    from polymap.training.trainer import score_events, train
    from tests.test_edges import make_trades

    trades = make_trades(120, 7)
    edges = build_temporal_edges(trades, window_seconds=300)
    bundle = build_graph(edges, build_wallet_features(trades))
    split = temporal_split(bundle, 0.7)
    cfg = ModelConfig(epochs=1, batch_size=32, eval_batch_size=8,
                      eval_negatives=4, device="cpu", pair_features=True)
    result = train(bundle, split, cfg, negative="hard")
    sampler = build_sampler("hard", bundle.edge_index, split.train_idx, bundle.num_nodes)
    scored = score_events(result, bundle, split, sampler, n_negatives=4, batch_size=8)
    assert scored["positive"].shape == (len(split.test_idx),)
    assert scored["negative"].shape == (len(split.test_idx), 4)
