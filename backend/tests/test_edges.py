"""The vectorised edge builder must match the original loop exactly.

This test is the reason the rewrite is safe to trust. `naive_build_edges` is a
literal transcription of the notebook implementation; if the fast path ever
drifts from it, this fails.
"""
import numpy as np
import pandas as pd
import pytest

from polymap.features.edges import build_temporal_edges, build_edges_multi_market

KEY = ["source", "target", "timestamp", "time_diff", "same_side",
       "same_outcome", "size_ratio", "price_diff"]


def naive_build_edges(trades_df, window_seconds=300):
    trades = trades_df.sort_values(["timestamp", "wallet"],
                                   kind="mergesort").reset_index(drop=True)
    edges = []
    for i in range(len(trades)):
        a = trades.iloc[i]
        for j in range(i + 1, len(trades)):
            b = trades.iloc[j]
            td = b["timestamp"] - a["timestamp"]
            if td > window_seconds:
                break
            if a["wallet"] == b["wallet"]:
                continue
            sa, sb = float(a["size"]), float(b["size"])
            edges.append({
                "source": a["wallet"], "target": b["wallet"],
                "timestamp": b["timestamp"], "time_diff": float(td),
                "same_side": int(a["side"] == b["side"]),
                "same_outcome": int(a["outcome"] == b["outcome"]),
                "price_diff": abs(float(a["price"]) - float(b["price"])),
                "size_ratio": min(sa, sb) / max(sa, sb) if max(sa, sb) > 0 else 0.0,
            })
    return pd.DataFrame(edges)


def make_trades(n, seed=0, market="m1", span=None):
    rng = np.random.default_rng(seed)
    span = span or max(n // 3, 10)
    return pd.DataFrame({
        "market_id": market,
        "wallet": [f"0x{w:04d}" for w in rng.integers(0, max(n // 4, 2), n)],
        "token_id": "t", "outcome": rng.choice(["Yes", "No"], n),
        "side": rng.choice(["BUY", "SELL"], n),
        "size": rng.gamma(2, 50, n), "price": rng.uniform(0.01, 0.99, n).round(3),
        "timestamp": np.sort(rng.integers(0, span, n).astype(np.int64)),
        "tx_hash": "0x0",
    })


@pytest.mark.parametrize("n,seed", [(120, 0), (300, 1), (600, 2)])
def test_matches_naive(n, seed):
    trades = make_trades(n, seed)
    fast = build_temporal_edges(trades).sort_values(KEY).reset_index(drop=True)
    slow = naive_build_edges(trades).sort_values(KEY).reset_index(drop=True)
    assert len(fast) == len(slow)
    for col in KEY:
        if fast[col].dtype.kind == "f":
            np.testing.assert_allclose(fast[col].values, slow[col].values, rtol=1e-9)
        else:
            assert (fast[col].values == slow[col].values).all(), col


def test_no_self_loops():
    e = build_temporal_edges(make_trades(400, 3))
    assert (e["source"] != e["target"]).all()


def test_window_respected():
    e = build_temporal_edges(make_trades(400, 4), window_seconds=60)
    assert e["time_diff"].max() <= 60
    # 0 is valid: simultaneous trades are linked, matching the original loop.
    assert (e["time_diff"] >= 0).all()


def test_empty_input():
    empty = make_trades(10).iloc[0:0]
    assert build_temporal_edges(empty).empty


def test_markets_never_joined():
    a, b = make_trades(200, 5, "m1"), make_trades(200, 6, "m2")
    both = pd.concat([a, b], ignore_index=True)
    e = build_edges_multi_market(both)
    m1_wallets = set(a["wallet"])
    for market, group in e.groupby("market_id"):
        pool = m1_wallets if market == "m1" else set(b["wallet"])
        assert set(group["source"]) <= pool and set(group["target"]) <= pool


def test_single_market_guard():
    both = pd.concat([make_trades(50, 7, "m1"), make_trades(50, 8, "m2")],
                     ignore_index=True)
    with pytest.raises(ValueError, match="one market"):
        build_temporal_edges(both)


def test_fanout_cap_reduces_edges():
    trades = make_trades(500, 9, span=20)     # dense burst
    full = build_temporal_edges(trades)
    capped = build_temporal_edges(trades, max_pairs_per_trade=5)
    assert len(capped) < len(full)


def test_crowd_filter_removes_bursts():
    trades = make_trades(400, 10, span=60)
    assert len(build_temporal_edges(trades, drop_crowd_timestamps=5)) < \
           len(build_temporal_edges(trades))


def test_crowd_filter_removing_everything_is_not_a_crash():
    # A market whose entire history is one burst should return zero edges
    # rather than raising: that is a finding, not a failure.
    trades = make_trades(400, 11, span=4)
    assert build_temporal_edges(trades, drop_crowd_timestamps=3).empty
