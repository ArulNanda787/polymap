"""Validate TGNN candidate pairs against normalized on-chain funding transfers.

Funding links are observable graph edges, not proof of ownership. The shared-funder
measure can be dominated by high-fan-out sources (e.g. exchanges or protocol
contracts), so the validator supports excluding sources that fund more than a
configurable number of wallets in the analysis universe.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DISCLAIMER = (
    "Funding links are evidence of observed transfers or shared funding sources, "
    "not proof of common ownership, control, intent, or coordination."
)


def _pair_key(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = a.astype(str).str.lower(), b.astype(str).str.lower()
    return pd.Series(np.where(a <= b, a + "|" + b, b + "|" + a), index=a.index)


def load_transfers(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        t = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        t = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    else:
        raise ValueError("Funding input must be .csv, .tsv, or .parquet")
    required = {"from_wallet", "to_wallet", "timestamp"}
    missing = required - set(t.columns)
    if missing:
        raise ValueError(f"Funding file missing required columns: {sorted(missing)}")
    t = t.copy()
    t["from_wallet"] = t["from_wallet"].astype(str).str.strip().str.lower()
    t["to_wallet"] = t["to_wallet"].astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(t["timestamp"], errors="coerce")
    if numeric.notna().mean() > 0.9:
        t["timestamp"] = numeric.astype("int64")
    else:
        parsed = pd.to_datetime(t["timestamp"], utc=True, errors="coerce")
        t["timestamp"] = (parsed.astype("int64") // 10**9).where(parsed.notna(), np.nan)
        t = t.dropna(subset=["timestamp"])
        t["timestamp"] = t["timestamp"].astype("int64")
    t = t[(t["from_wallet"] != "") & (t["to_wallet"] != "")]
    t = t[t["from_wallet"] != t["to_wallet"]]
    # Address-scoped collection may contain the same transfer more than once.
    return t.drop_duplicates(subset=["from_wallet", "to_wallet", "timestamp"])


def _funding_evidence(
    transfers: pd.DataFrame,
    wallet_universe: set[str] | None = None,
    max_funder_fanout: int = 100,
) -> tuple[set[str], set[str], dict[str, set[str]], set[str], dict[str, int]]:
    """Build pair evidence; shared-funder counts use only analysis wallets.

    A source is excluded from shared-funder evidence when it funded more than
    max_funder_fanout distinct wallets in wallet_universe. Set threshold <= 0
    to disable this filter. Direct transfers are not affected by this filter.
    """
    t = transfers.copy()
    t["from_wallet"] = t["from_wallet"].astype(str).str.lower()
    t["to_wallet"] = t["to_wallet"].astype(str).str.lower()
    if wallet_universe is not None:
        universe = {str(w).lower() for w in wallet_universe}
        t_recipients = t[t["to_wallet"].isin(universe)]
    else:
        universe = set(t["to_wallet"]) | set(t["from_wallet"])
        t_recipients = t

    direct = set(_pair_key(t["from_wallet"], t["to_wallet"]))
    recipients: dict[str, set[str]] = {
        str(source): set(group["to_wallet"].astype(str))
        for source, group in t_recipients.groupby("from_wallet")
    }
    fanout = {source: len(wallets) for source, wallets in recipients.items()}
    excluded = {
        source for source, count in fanout.items()
        if max_funder_fanout > 0 and count > max_funder_fanout
    }

    shared: set[str] = set()
    for source, wallets in recipients.items():
        if source in excluded or len(wallets) < 2:
            continue
        w = sorted(wallets)
        for i, a in enumerate(w):
            for b in w[i + 1:]:
                shared.add(f"{a}|{b}")
    return direct, shared, recipients, excluded, fanout


def _wallet_label_permutation_p(
    transfers: pd.DataFrame,
    candidate_pairs: list[str],
    control_pairs: list[str],
    *,
    seed: int = 43,
    n_perm: int = 2000,
) -> float | None:
    """Permutation test that preserves wallet overlap and transfer-graph topology.

    Wallet labels are permuted on the induced transfer graph for wallets appearing
    in the tested pairs. This preserves the graph's degree/topology and the
    candidate/control pair overlap pattern, unlike shuffling pair outcomes as if
    each pair were independent. It is a descriptive test, not a causal test.
    """
    if n_perm < 1 or not candidate_pairs or not control_pairs:
        return None

    def parse_pair(pair: str) -> tuple[str, str]:
        a, b = pair.split("|", 1)
        return a, b

    wallet_names = sorted({
        w for pair in candidate_pairs + control_pairs for w in parse_pair(pair)
    })
    if len(wallet_names) < 2:
        return None
    wallet_to_idx = {w: i for i, w in enumerate(wallet_names)}

    t = transfers.copy()
    t["from_wallet"] = t["from_wallet"].astype(str).str.lower()
    t["to_wallet"] = t["to_wallet"].astype(str).str.lower()
    t = t[t["from_wallet"].isin(wallet_to_idx) & t["to_wallet"].isin(wallet_to_idx)]
    if t.empty:
        return None

    # Use unique undirected links because the measured feature is whether a
    # direct transfer occurred in either direction.
    edge_idx = {
        tuple(sorted((wallet_to_idx[a], wallet_to_idx[b])))
        for a, b in zip(t["from_wallet"], t["to_wallet"])
        if a != b
    }
    if not edge_idx:
        return None
    edges = np.asarray(sorted(edge_idx), dtype=np.int64)

    def pair_indices(pairs: list[str]) -> list[tuple[int, int]]:
        return [
            tuple(sorted((wallet_to_idx[a], wallet_to_idx[b])))
            for a, b in map(parse_pair, pairs)
        ]

    cand_idx = pair_indices(candidate_pairs)
    ctrl_idx = pair_indices(control_pairs)
    observed_edges = set(map(tuple, edges.tolist()))
    observed = abs(
        sum(p in observed_edges for p in cand_idx) / len(cand_idx)
        - sum(p in observed_edges for p in ctrl_idx) / len(ctrl_idx)
    )

    rng = np.random.default_rng(seed)
    exceed = 0
    n_wallets = len(wallet_names)
    for _ in range(n_perm):
        # mapping old node labels to randomly assigned node labels
        mapping = rng.permutation(n_wallets)
        perm_edges = {
            tuple(sorted((int(mapping[a]), int(mapping[b]))))
            for a, b in edges
        }
        delta = abs(
            sum(p in perm_edges for p in cand_idx) / len(cand_idx)
            - sum(p in perm_edges for p in ctrl_idx) / len(ctrl_idx)
        )
        exceed += delta >= observed - 1e-12
    return float((exceed + 1) / (n_perm + 1))


def validate_funding(
    candidates: pd.DataFrame,
    edges: pd.DataFrame,
    transfers: pd.DataFrame,
    *,
    controls_per_candidate: int = 3,
    seed: int = 42,
    covered_wallets: set[str] | None = None,
    max_funder_fanout: int = 100,
    n_permutations: int = 2000,
) -> tuple[pd.DataFrame, dict]:
    """Return candidate/control pair-level evidence and a compact comparison."""
    if controls_per_candidate < 1:
        raise ValueError("controls_per_candidate must be >= 1")
    if max_funder_fanout < 0:
        raise ValueError("max_funder_fanout must be >= 0 (0 disables filtering)")
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    for frame, cols, label in [
        (candidates, {"wallet_a", "wallet_b"}, "candidate pairs"),
        (edges, {"source", "target"}, "temporal edges"),
        (transfers, {"from_wallet", "to_wallet", "timestamp"}, "funding transfers"),
    ]:
        if not cols.issubset(frame.columns):
            raise ValueError(f"{label} missing columns: {sorted(cols - set(frame.columns))}")

    c = candidates.copy()
    c["wallet_a"] = c["wallet_a"].astype(str).str.lower()
    c["wallet_b"] = c["wallet_b"].astype(str).str.lower()
    c = c[c.wallet_a != c.wallet_b].copy()
    c["pair"] = _pair_key(c.wallet_a, c.wallet_b)
    e = edges.copy()
    e["pair"] = _pair_key(e.source, e.target)
    counts = e.groupby("pair").size().rename("cotiming_count").reset_index()
    all_pairs = counts.copy()
    all_pairs[["wallet_a", "wallet_b"]] = all_pairs["pair"].str.split("|", n=1, expand=True)
    all_pairs = all_pairs[all_pairs.wallet_a != all_pairs.wallet_b]
    all_pairs["log_cotiming_count"] = np.log1p(all_pairs["cotiming_count"])
    candidate_keys = set(c["pair"])
    c = c.merge(counts, on="pair", how="left")
    c["cotiming_count"] = c["cotiming_count"].fillna(0).astype(int)

    pool = all_pairs[~all_pairs.pair.isin(candidate_keys)].copy()
    n_pool_before = len(pool)
    if covered_wallets is not None:
        cov = {w.lower() for w in covered_wallets}
        pool = pool[pool.wallet_a.isin(cov) & pool.wallet_b.isin(cov)].copy()

    # Fan-out is measured over wallets in the analysis universe, not all possible
    # recipients on-chain. This aligns the filter with the tested population.
    wallet_universe = set(c.wallet_a) | set(c.wallet_b)
    if not pool.empty:
        wallet_universe |= set(pool.wallet_a) | set(pool.wallet_b)
    direct, shared, recipients, excluded_sources, fanout = _funding_evidence(
        transfers, wallet_universe, max_funder_fanout
    )

    rng = np.random.default_rng(seed)
    chosen: set[str] = set()
    rows: list[dict] = []

    def evidence(pair: str) -> tuple[bool, bool]:
        return pair in direct, pair in shared

    for row in c.itertuples(index=False):
        pair = row.pair
        d, s = evidence(pair)
        rows.append({
            "pair": pair, "wallet_a": row.wallet_a, "wallet_b": row.wallet_b,
            "group": "tgnn_candidate", "cotiming_count": int(row.cotiming_count),
            "match_abs_log_cotiming_diff": 0.0, "direct_transfer": d,
            "shared_funder": s, "any_funding_link": d or s,
            "tgnn_score": getattr(row, "mean_positive_score", np.nan),
            "evidence_label": "candidate relationship; not ownership",
        })
        if pool.empty:
            continue
        target = np.log1p(max(int(row.cotiming_count), 0))
        available = pool[~pool.pair.isin(chosen)].copy()
        if available.empty:
            available = pool.copy()
        available["_diff"] = (available.log_cotiming_count - target).abs()
        best_diff = available["_diff"].min()
        nearest = available[available["_diff"] <= best_diff + 1e-12]
        n_take = min(controls_per_candidate, len(nearest))
        sampled_idx = rng.choice(nearest.index.to_numpy(), size=n_take, replace=False)
        for idx in sampled_idx:
            ctrl = available.loc[idx]
            chosen.add(ctrl.pair)
            cd, cs = evidence(ctrl.pair)
            rows.append({
                "pair": ctrl.pair, "wallet_a": ctrl.wallet_a, "wallet_b": ctrl.wallet_b,
                "group": "cotiming_matched_control",
                "cotiming_count": int(ctrl.cotiming_count),
                "match_abs_log_cotiming_diff": float(ctrl["_diff"]),
                "direct_transfer": cd, "shared_funder": cs,
                "any_funding_link": cd or cs, "tgnn_score": np.nan,
                "evidence_label": "matched control; not an ownership assessment",
            })

    out = pd.DataFrame(rows)
    cand = out[out.group == "tgnn_candidate"]
    ctrl = out[out.group == "cotiming_matched_control"]

    def rate(frame: pd.DataFrame, col: str) -> float | None:
        return float(frame[col].mean()) if len(frame) else None

    def diff_summary(col: str) -> dict:
        a, b = rate(cand, col), rate(ctrl, col)
        return {
            "candidate_rate": a, "matched_control_rate": b,
            "risk_difference": (a - b) if a is not None and b is not None else None,
            "candidate_n": int(len(cand)), "control_n": int(len(ctrl)),
        }

    # Retain the historical pair-level p-value for any_funding_link for
    # comparability, but label it explicitly as pair-independent/descriptive.
    rng_p = np.random.default_rng(seed + 1)
    def pair_permutation_p(col: str, n_perm: int = 10000) -> float | None:
        vals = out[col].astype(bool).to_numpy()
        n_c, n_ctrl = len(cand), len(ctrl)
        if n_c == 0 or n_ctrl == 0:
            return None
        observed = abs(cand[col].mean() - ctrl[col].mean())
        exceed = 0
        for _ in range(n_perm):
            perm = rng_p.permutation(vals)
            delta = abs(perm[:n_c].mean() - perm[n_c:].mean())
            exceed += delta >= observed - 1e-12
        return float((exceed + 1) / (n_perm + 1))

    cand_pair_list = cand["pair"].astype(str).tolist()
    ctrl_pair_list = ctrl["pair"].astype(str).tolist()
    wallet_perm_p = _wallet_label_permutation_p(
        transfers, cand_pair_list, ctrl_pair_list,
        seed=seed + 2, n_perm=n_permutations,
    )

    summary = {
        "objective": "Test whether TGNN candidates have more on-chain funding evidence than co-timing-matched controls.",
        "candidate_pairs": int(len(cand)), "matched_control_pairs": int(len(ctrl)),
        "controls_per_candidate_requested": controls_per_candidate,
        "control_pool_size": int(len(pool)),
        "control_pool_size_before_coverage_filter": int(n_pool_before),
        "controls_restricted_to_fetched_wallets": covered_wallets is not None,
        "n_wallets_with_onchain_data": None if covered_wallets is None else len(covered_wallets),
        "mean_abs_log_cotiming_match_difference": float(ctrl.match_abs_log_cotiming_diff.mean()) if len(ctrl) else None,
        "max_funder_fanout": max_funder_fanout,
        "funder_fanout_threshold_semantics": (
            "Exclude a source from shared-funder evidence if it funds more than this "
            "many distinct wallets in the analysis universe; 0 disables exclusion."
        ),
        "excluded_shared_funder_count": len(excluded_sources),
        "excluded_shared_funders_top20": sorted(
            ((source, int(fanout[source])) for source in excluded_sources),
            key=lambda item: item[1], reverse=True
        )[:20],
        "direct_transfer": diff_summary("direct_transfer"),
        "shared_funder": diff_summary("shared_funder"),
        "any_funding_link": diff_summary("any_funding_link"),
        "permutation_p_any_funding_link_pair_independent_descriptive": pair_permutation_p("any_funding_link"),
        "permutation_p_direct_transfer_wallet_label": wallet_perm_p,
        "wallet_label_permutation": {
            "n_permutations": n_permutations,
            "method": "Permute wallet labels on the induced transfer graph, preserving transfer topology and pair overlap.",
            "interpretation": "Descriptive association test; not causal evidence or proof of ownership.",
        },
        "plain_cotiming_baseline": "cotiming_count is reported for candidates and matched controls; compare this simple count baseline with TGNN held-out ranking metrics separately.",
        "interpretation": DISCLAIMER,
        "limitations": [
            "Funding coverage is limited to the supplied chain, assets, addresses, and time window.",
            "Shared-funder links can be caused by exchanges, bridges, routers, or custodial services.",
            "High-fan-out sources are excluded from shared-funder evidence, but the threshold should be sensitivity-tested.",
            "Controls use nearest log1p co-timing count; inspect matching distance and sample sizes.",
            "The wallet-label permutation is descriptive and does not establish causality or ownership.",
        ],
    }
    return out, summary


def run_funding_validation(
    candidate_path: str | Path,
    edge_path: str | Path,
    funding_path: str | Path,
    output_dir: str | Path,
    controls_per_candidate: int = 3,
    seed: int = 42,
    max_funder_fanout: int = 100,
    n_permutations: int = 2000,
) -> dict:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(candidate_path)
    edges = pd.read_parquet(edge_path)
    transfers = load_transfers(funding_path)
    covered = None
    if "queried_wallet" in transfers.columns:
        covered = set(transfers["queried_wallet"].astype(str).str.lower())
    pairs, summary = validate_funding(
        candidates, edges, transfers,
        controls_per_candidate=controls_per_candidate, seed=seed,
        covered_wallets=covered, max_funder_fanout=max_funder_fanout,
        n_permutations=n_permutations,
    )
    if covered is not None and summary["control_pool_size"] == 0:
        raise ValueError(
            "no control pairs have both wallets fetched on-chain; fetch more wallets "
            "(e.g. a lower --min-events) or add non-candidate wallets to the list"
        )
    pairs.to_parquet(outdir / "funding_validation_pairs.parquet", index=False)
    (outdir / "funding_validation.json").write_text(json.dumps(summary, indent=2))
    return summary
