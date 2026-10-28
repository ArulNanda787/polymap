import pandas as pd

from polymap.analysis.funding_validation import (
    _funding_evidence,
    _wallet_label_permutation_p,
    validate_funding,
)


def test_high_fanout_source_is_excluded_from_shared_funder_evidence():
    transfers = pd.DataFrame({
        "from_wallet": ["hub", "hub", "hub", "small", "small"],
        "to_wallet": ["a", "b", "c", "a", "b"],
        "timestamp": [1, 2, 3, 4, 5],
    })
    direct, shared, recipients, excluded, fanout = _funding_evidence(
        transfers, {"a", "b", "c"}, max_funder_fanout=2
    )
    assert "hub" in excluded
    assert fanout["hub"] == 3
    assert "a|b" in shared  # pair remains shared via small funder
    assert "a|c" not in shared
    assert "a|hub" in direct


def test_zero_fanout_threshold_disables_filter():
    transfers = pd.DataFrame({
        "from_wallet": ["hub", "hub"],
        "to_wallet": ["a", "b"],
        "timestamp": [1, 2],
    })
    _, shared, _, excluded, _ = _funding_evidence(
        transfers, {"a", "b"}, max_funder_fanout=0
    )
    assert not excluded
    assert "a|b" in shared


def test_wallet_label_permutation_is_reproducible_and_bounded():
    transfers = pd.DataFrame({
        "from_wallet": ["a", "c"],
        "to_wallet": ["b", "d"],
        "timestamp": [1, 2],
    })
    p1 = _wallet_label_permutation_p(
        transfers, ["a|b", "a|c"], ["c|d", "b|d"], seed=11, n_perm=100
    )
    p2 = _wallet_label_permutation_p(
        transfers, ["a|b", "a|c"], ["c|d", "b|d"], seed=11, n_perm=100
    )
    assert p1 == p2
    assert p1 is not None and 0 < p1 <= 1


def test_validation_reports_fanout_and_wallet_permutation():
    candidates = pd.DataFrame({"wallet_a": ["a"], "wallet_b": ["b"]})
    edges = pd.DataFrame({
        "source": ["a", "b", "c", "d"],
        "target": ["b", "a", "d", "c"],
    })
    transfers = pd.DataFrame({
        "from_wallet": ["hub", "hub", "a"],
        "to_wallet": ["a", "b", "b"],
        "timestamp": [1, 2, 3],
    })
    pairs, summary = validate_funding(
        candidates, edges, transfers, controls_per_candidate=1,
        seed=5, max_funder_fanout=1, n_permutations=50,
    )
    assert summary["max_funder_fanout"] == 1
    assert summary["excluded_shared_funder_count"] >= 1
    assert "permutation_p_direct_transfer_wallet_label" in summary
    assert set(pairs["group"]) <= {"tgnn_candidate", "cotiming_matched_control"}
