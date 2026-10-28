"""Funding robustness: coverage, anchor-fixed controls, hub sweeps, deposit-only."""
import numpy as np
import pandas as pd
import pytest

from polymap.analysis.funding_robustness import (FundingEvidence, build_pairs, evaluate,
                                                 pair_key, ZERO_ADDRESS)

W = [f"0x{i:040x}" for i in range(1, 31)]
HUB = "0x" + "ab" * 20
REAL = "0x" + "cd" * 20


def _transfers(rows):
    df = pd.DataFrame(rows, columns=["from_wallet", "to_wallet", "timestamp", "token_symbol", "queried_wallet"])
    df["tx_hash"] = [f"h{i}" for i in range(len(df))]
    return df


def _edges(pairs, reps=5):
    rows = []
    for a, b in pairs:
        rows += [{"source": a, "target": b}] * reps
    return pd.DataFrame(rows)


def test_hub_is_ignored_by_fanout_cap_but_rare_funder_counts():
    rows = [(HUB, w, 100, "pUSD", w) for w in W]                    # hub funds everyone
    rows += [(REAL, W[0], 100, "pUSD", W[0]), (REAL, W[1], 100, "pUSD", W[1])]
    ev = FundingEvidence(_transfers(rows), pd.Series({w: 200 for w in W}))
    assert ev.has_shared(W[0], W[1], "any", 100) is True            # both hub and REAL
    assert ev.shared(W[0], W[1], "any", 5) == {REAL}                # hub dropped at cap 5
    assert ev.shared(W[2], W[3], "any", 5) == set()                 # only the hub in common
    assert ev.rarity_score(W[0], W[1]) > ev.rarity_score(W[2], W[3])


def test_deposit_only_requires_stablecoin_before_first_trade_and_ignores_zero_address():
    rows = [(REAL, W[0], 100, "pUSD", W[0]), (REAL, W[1], 100, "pUSD", W[1]),   # before first trade
            ("0x" + "ee" * 20, W[0], 100, "POL", W[0]), ("0x" + "ee" * 20, W[1], 100, "POL", W[1]),  # gas
            ("0x" + "ff" * 20, W[2], 900, "pUSD", W[2]), ("0x" + "ff" * 20, W[3], 900, "pUSD", W[3]),  # after trade
            (ZERO_ADDRESS, W[4], 100, "pUSD", W[4]), (ZERO_ADDRESS, W[5], 100, "pUSD", W[5])]
    ft = pd.Series({w: 500 for w in W})
    ev = FundingEvidence(_transfers(rows), ft)
    assert ev.has_shared(W[0], W[1], "deposit", 100)
    assert ev.shared(W[0], W[1], "deposit", 100) == {REAL}          # POL gas excluded
    assert not ev.has_shared(W[2], W[3], "deposit", 100)            # funded after first trade
    assert not ev.has_shared(W[4], W[5], "any", 100)                # mint address ignored


def test_controls_share_the_anchor_are_unique_and_never_candidates():
    cand = pd.DataFrame({"wallet_a": [W[0], W[2]], "wallet_b": [W[1], W[3]]})
    pairs_real = [(W[0], W[1]), (W[2], W[3])] + [(W[0], w) for w in W[4:10]] + [(W[1], w) for w in W[10:16]] \
        + [(W[2], w) for w in W[16:22]] + [(W[3], w) for w in W[22:28]]
    out, info = build_pairs(cand, _edges(pairs_real), set(W), seed=1)
    ctrl = out[out.group == "control"]
    assert ctrl.pair.is_unique and info["controls_reused"] == 0
    assert not set(ctrl.pair) & {pair_key(W[0], W[1]), pair_key(W[2], W[3])}
    for r in ctrl.itertuples():
        assert r.anchor in (r.wallet_a, r.wallet_b)                 # one wallet fixed
    assert len(ctrl) == 4                                           # 2 per candidate


def test_candidates_without_coverage_are_dropped_not_counted_as_unlinked():
    cand = pd.DataFrame({"wallet_a": [W[0], W[0]], "wallet_b": [W[1], "0x" + "99" * 20]})
    out, info = build_pairs(cand, _edges([(W[0], W[1]), (W[0], W[5]), (W[1], W[6])]), {W[0], W[1], W[5], W[6]})
    assert info["candidate_pairs_dropped_no_coverage"] == 1
    assert (out.group == "candidate").sum() == 1


def test_planted_direct_transfer_effect_is_detected_with_interval():
    rng = np.random.default_rng(0)
    ws = W
    cand = [(ws[2 * i], ws[2 * i + 1]) for i in range(10)]
    ctrl_edges = [(ws[2 * i], ws[(2 * i + 5 + j) % 30]) for i in range(10) for j in range(3)]
    ed = _edges(cand + [p for p in ctrl_edges if pair_key(*p) not in {pair_key(*c) for c in cand}])
    rows = [(a, b, 100, "pUSD", b) for a, b in cand]               # every candidate has a direct transfer
    rows += [(REAL, w, 100, "pUSD", w) for w in ws]                 # keep every wallet "fetched"
    ev = FundingEvidence(_transfers(rows), pd.Series({w: 200 for w in ws}))
    pairs, _ = build_pairs(pd.DataFrame(cand, columns=["wallet_a", "wallet_b"]), ed, ev.covered, seed=2)
    t = evaluate(pairs, ev, caps=(10,), n_boot=200)
    d = t[t.measure.str.startswith("direct")].iloc[0]
    assert d.candidate == 1.0 and d.control == 0.0 and d.difference == 1.0
