"""Schema, features, graph, splits, metrics and screening."""
import numpy as np
import pandas as pd
import pytest

from polymap.analysis.relationships import run_screening
from polymap.config import Config, load_config
from polymap.data.schema import SchemaError, validate_trades
from polymap.data.trades import normalise_trades, parse_market
from polymap.evaluation.metrics import classification_metrics, ranking_metrics
from polymap.features.edges import build_temporal_edges
from polymap.features.wallets import build_wallet_features
from polymap.graph.build import build_graph
from polymap.training.negatives import build_sampler
from polymap.training.splits import temporal_split
from tests.test_edges import make_trades


def test_schema_rejects_missing_columns():
    with pytest.raises(SchemaError, match="missing columns"):
        validate_trades(pd.DataFrame({"wallet": ["0x1"]}))


def test_normalise_trades_uses_api_field_names():
    raw = [{"proxyWallet": "0xABC", "asset": "tok1", "side": "BUY", "size": 10,
            "price": 0.5, "timestamp": 1700000000, "outcome": "Yes",
            "transactionHash": "0xdead", "conditionId": "0xmkt"}]
    df = normalise_trades(raw, "0xmkt")
    assert df.loc[0, "wallet"] == "0xabc"          # lowercased
    assert df.loc[0, "token_id"] == "tok1"
    assert df.loc[0, "side"] == "BUY"


def test_normalise_trades_names_the_missing_field():
    with pytest.raises(ValueError, match="TRADE_FIELD_MAP"):
        normalise_trades([{"proxyWallet": "0x1"}], "0xmkt")


def test_parse_market_rejects_unresolved():
    base = {"conditionId": "0x1", "slug": "s", "question": "q",
            "outcomes": '["Yes","No"]', "clobTokenIds": '["a","b"]', "volume": "1000"}
    assert parse_market({**base, "outcomePrices": '["1","0"]'})["resolved"] is True
    assert parse_market({**base, "outcomePrices": '["0.55","0.45"]'})["resolved"] is False


def test_config_fingerprint_tracks_research_choices():
    a = Config()
    b = load_config(**{"graph": {"window_seconds": 60}})
    assert a.fingerprint() != b.fingerprint()
    assert Config(run_name="x").fingerprint() == Config(run_name="y").fingerprint()


def test_features_have_no_outcome_leakage():
    f = build_wallet_features(make_trades(300, 1))
    banned = {"won", "winner", "payout", "pnl", "profit", "resolved"}
    assert not any(b in c.lower() for c in f.columns for b in banned)


def test_graph_rejects_feature_mismatch():
    trades = make_trades(300, 2)
    edges = build_temporal_edges(trades)
    feats = build_wallet_features(trades)
    with pytest.raises(ValueError, match="no features"):
        build_graph(edges, feats.iloc[:2])


def test_split_is_chronological():
    trades = make_trades(800, 3)
    b = build_graph(build_temporal_edges(trades), build_wallet_features(trades))
    s = temporal_split(b, 0.7)
    assert b.timestamps[s.train_idx].max() <= b.timestamps[s.test_idx].min()
    assert len(s.train_idx) and len(s.test_idx)


def test_hard_sampler_draws_from_history():
    trades = make_trades(600, 4)
    b = build_graph(build_temporal_edges(trades), build_wallet_features(trades))
    s = temporal_split(b, 0.7)
    sampler = build_sampler("hard", b.edge_index, s.train_idx, b.num_nodes)
    rng = np.random.default_rng(0)
    src = int(b.edge_index[0][s.train_idx[0]])
    assert sampler.sample(src, rng) in sampler.history[src]


def test_below_chance_auc_is_flagged():
    rng = np.random.default_rng(0)
    m = classification_metrics(rng.normal(0, 1, 300), rng.normal(3, 1, 300))
    assert m["roc_auc"] < 0.5 and m["below_chance"] is True


def test_ranking_metrics_bounds():
    rng = np.random.default_rng(0)
    m = ranking_metrics(rng.normal(5, 1, 200), rng.normal(0, 1, (200, 10)))
    assert 0 < m["mrr"] <= 1 and m["hits@1"] <= m["hits@5"] <= m["hits@10"]


def test_screening_categories_are_exhaustive():
    rng = np.random.default_rng(0)
    n = 900
    ev = pd.DataFrame({
        "source": [f"0x{i:02d}" for i in rng.integers(0, 25, n)],
        "target": [f"0x{i:02d}" for i in rng.integers(0, 25, n)],
        "timestamp": np.sort(rng.integers(0, 30 * 86400, n)),
        "positive_score": rng.normal(5, 2, n), "negative_score": rng.normal(1, 2, n)})
    ev["margin"] = ev["positive_score"] - ev["negative_score"]
    ed = ev[["source", "target", "timestamp"]].copy()
    for c, v in [("same_side", rng.integers(0, 2, n)), ("same_outcome", rng.integers(0, 2, n)),
                 ("same_behavior", rng.integers(0, 2, n)), ("time_diff", rng.exponential(60, n)),
                 ("price_diff", rng.exponential(0.05, n)), ("size_ratio", rng.uniform(0, 1, n))]:
        ed[c] = v
    out = run_screening(ev, ed)
    assert set(out["category"]) <= {"persistent_tight", "persistent_loose",
                                    "one_off_tight", "one_off_loose"}
    assert (out["n_events"] >= 3).all()
