"""Model-free candidate screen, vectorised samplers, batched trainer."""
import numpy as np
import pandas as pd
import pytest

from polymap.analysis.relationships import (candidate_wallets, prescreen_pairs,
                                            select_candidates)
from polymap.config import ModelConfig, ScreeningConfig
from polymap.features.edges import build_temporal_edges
from polymap.features.wallets import build_wallet_features
from polymap.graph.build import build_graph
from polymap.training.negatives import build_sampler
from polymap.training.splits import temporal_split
from tests.test_edges import make_trades


def _edges(n=600, seed=3):
    return build_temporal_edges(make_trades(n, seed), window_seconds=300)


def test_prescreen_needs_no_model_and_respects_min_events():
    e = _edges()
    out = prescreen_pairs(e, ScreeningConfig(min_events=3))
    assert len(out) > 0
    assert (out["real_events"] >= 3).all()
    assert set(out["category"]) <= {"persistent_tight", "persistent_loose",
                                    "one_off_tight", "one_off_loose"}
    # undirected: no pair appears twice
    assert out["pair"].is_unique


def test_select_candidates_filters_and_caps():
    out = prescreen_pairs(_edges(), ScreeningConfig(min_events=2))
    sel = select_candidates(out, ("one_off_tight", "persistent_tight"), max_pairs=0)
    assert set(sel["category"]) <= {"one_off_tight", "persistent_tight"}
    capped = select_candidates(out, ("one_off_tight", "persistent_tight", "one_off_loose"), 5)
    assert len(capped) <= 5
    wallets = candidate_wallets(capped)
    assert wallets == sorted(set(wallets))


def test_vector_samplers_match_single_samplers_in_support():
    e = _edges()
    b = build_graph(e, build_wallet_features(make_trades(600, 3)))
    s = temporal_split(b, 0.7)
    rng = np.random.default_rng(0)
    srcs = b.edge_index[0][s.test_idx[:200]]
    for kind in ("random", "hard", "popularity"):
        sampler = build_sampler(kind, b.edge_index, s.train_idx, b.num_nodes)
        out = sampler.sample_many(srcs, rng, 7)
        assert out.shape == (200, 7)
        assert out.min() >= 0 and out.max() < b.num_nodes
    hard = build_sampler("hard", b.edge_index, s.train_idx, b.num_nodes)
    out = hard.sample_many(srcs, rng, 5)
    for row, src in zip(out, srcs):
        if hard.history.get(int(src)):          # warm sources draw from history
            assert set(row.tolist()) <= set(hard.history[int(src)])


def test_batched_trainer_learns_and_scores_every_test_event():
    pytest.importorskip("torch")
    from polymap.training.trainer import score_events, train
    e = _edges(800, 5)
    b = build_graph(e, build_wallet_features(make_trades(800, 5)))
    s = temporal_split(b, 0.7)
    cfg = ModelConfig(epochs=2, batch_size=64, device="cpu")
    r = train(b, s, cfg, "random")
    assert len(r.losses) > 0 and np.isfinite(r.losses).all()
    sampler = build_sampler("random", b.edge_index, s.train_idx, b.num_nodes)
    sc = score_events(r, b, s, sampler, n_negatives=10, batch_size=4)
    assert len(sc["positive"]) == len(s.test_idx)
    assert sc["negative"].shape == (len(s.test_idx), 10)
    ev = sc["events"]
    assert list(ev["edge_idx"]) == list(s.test_idx)
    assert np.allclose(ev["margin"], ev["positive_score"] - ev["negative_score"])


def test_eval_gap_matches_naive_last_seen():
    pytest.importorskip("torch")
    from polymap.training.trainer import score_events, train
    e = _edges(500, 9)
    b = build_graph(e, build_wallet_features(make_trades(500, 9)))
    s = temporal_split(b, 0.7)
    r = train(b, s, ModelConfig(epochs=1, batch_size=64, device="cpu"), "random")
    sampler = build_sampler("random", b.edge_index, s.train_idx, b.num_nodes)
    ev = score_events(r, b, s, sampler, n_negatives=5)["events"]
    src, tgt = b.edge_index
    ts = b.timestamps
    last = {}
    for i in s.train_idx:
        last[(min(src[i], tgt[i]), max(src[i], tgt[i]))] = int(ts[i])
    expected = []
    for i in s.test_idx:
        k = (min(src[i], tgt[i]), max(src[i], tgt[i]))
        expected.append(int(ts[i]) - last[k] if k in last else np.nan)
        last[k] = int(ts[i])
    np.testing.assert_allclose(ev["gap_s"].to_numpy(), np.array(expected, dtype=float))


def test_funding_controls_only_come_from_fetched_wallets():
    from polymap.analysis.funding_validation import validate_funding
    e = _edges(600, 3)
    pairs = prescreen_pairs(e, ScreeningConfig(min_events=2))
    cand = select_candidates(pairs, ("one_off_tight", "persistent_tight", "persistent_loose",
                                     "one_off_loose"), max_pairs=6)
    covered = set(candidate_wallets(cand)) | set(pairs["wallet_a"].head(30)) | set(pairs["wallet_b"].head(30))
    transfers = pd.DataFrame({"from_wallet": ["0xf"] * 2, "to_wallet": [cand.wallet_a[0], cand.wallet_b[0]],
                              "timestamp": [1, 2]})
    out, summ = validate_funding(cand, e.rename(columns={}), transfers, covered_wallets=covered)
    ctrl = out[out.group == "cotiming_matched_control"]
    assert len(ctrl) > 0
    assert ctrl.wallet_a.isin(covered).all() and ctrl.wallet_b.isin(covered).all()
    assert summ["controls_restricted_to_fetched_wallets"] is True
    assert summ["shared_funder"]["candidate_rate"] > 0     # the planted shared funder is found
