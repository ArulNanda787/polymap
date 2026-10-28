import numpy as np
import pandas as pd

from polymap.analysis.null_model import MODES, count_tight_pairs, make_null_trades
from tests.test_edges import make_trades


def test_nulls_preserve_what_they_claim_to():
    t = make_trades(400, 1)
    rng = np.random.default_rng(0)
    w = make_null_trades(t, "wallet", rng)
    assert sorted(w["timestamp"]) == sorted(t["timestamp"])      # tape untouched
    assert sorted(w["wallet"]) == sorted(t["wallet"])            # same trade counts
    ts = make_null_trades(t, "timestamp", rng)
    assert sorted(ts["timestamp"]) == sorted(t["timestamp"])
    assert (ts["wallet"].to_numpy() == t["wallet"].to_numpy()).all()
    sh = make_null_trades(t, "shift", rng)
    assert sh["timestamp"].between(t["timestamp"].min(), t["timestamp"].max()).all()
    assert set(MODES) == {"wallet", "stratified", "stratified_side", "shift", "timestamp"}


def test_count_tight_pairs_hand_example():
    # pair (a,b): 3 edges, median gaps 1s / 0.0 -> tight; pair (a,c): 3 edges, 100s -> not
    rows = []
    for _ in range(3):
        rows.append(("m", "a", "b", 1.0, 0.0))
        rows.append(("m", "c", "a", 100.0, 0.0))
    e = pd.DataFrame(rows, columns=["market_id", "source", "target", "time_diff", "price_diff"])
    r = count_tight_pairs(e, min_events=3, max_time_diff=5, max_price_diff=0.01)
    assert r["n_pairs"] == 2 and r["n_tight"] == 1 and r["n_tight_timing_only"] == 1
    assert count_tight_pairs(e, 4, 5, 0.01)["n_pairs"] == 0