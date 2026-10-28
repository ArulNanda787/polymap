"""Funding evidence, redone so the candidate/control comparison is fair.

Why this exists
---------------
The first validator drew controls from any non-candidate pair and counted a
"shared funder" if any address had sent anything to both wallets. Two problems:

1. A handful of protocol/exchange addresses fund almost every wallet, so the
   shared-funder rate saturates near 100% for everyone and says nothing.
2. Controls were matched only on co-timing count, so they can differ from the
   candidates in ways unrelated to the question (how active the wallets are).

What this module does differently
---------------------------------
* **Controls keep one wallet fixed.** For a candidate pair (A, B) the controls
  are (A, C) and (B, D): same anchor wallet, a different partner who also
  traded alongside it, matched on co-timing count, never reused. The anchor's
  own on-chain history is therefore identical on both sides of the comparison.
* **Coverage is enforced.** Only wallets whose on-chain history was fetched
  (`queried_wallet`) can be in a candidate pair or a control pair.
* **Hubs are handled by a sweep, not one threshold.** Every shared-funder
  measure is reported at several fan-out caps (a funder that reaches more than
  `cap` fetched wallets is ignored), plus a rarity-weighted score.
* **Deposit-only definition.** Stablecoin transfers received at or before the
  wallet's first trade in this market, from non-hub senders. This removes
  trading settlement flows and gas top-ups.
* **Uncertainty.** A wallet-level bootstrap gives a 95% interval for each
  candidate-minus-control difference, because pairs share wallets and are not
  independent.

Everything here is descriptive. A funding link is not proof of common
ownership: exchanges, bridges and routers create shared funders for unrelated
wallets.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ZERO_ADDRESS = "0x" + "0" * 40
STABLE_SYMBOLS = ("pUSD", "USDC.e", "USDC", "USDT0")
DEFAULT_CAPS = (5, 10, 20, 50, 100)

DISCLAIMER = (
    "Funding links are evidence of observed transfers or shared funding sources, "
    "not proof of common ownership, control, intent, or coordination."
)


# --------------------------------------------------------------------- inputs
def pair_key(a: str, b: str) -> str:
    return f"{a}|{b}" if a <= b else f"{b}|{a}"


def load_transfers(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    t = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
    need = {"from_wallet", "to_wallet", "timestamp", "queried_wallet"}
    if not need <= set(t.columns):
        raise ValueError(f"transfers file missing columns: {sorted(need - set(t.columns))}")
    for c in ("from_wallet", "to_wallet", "queried_wallet"):
        t[c] = t[c].astype(str).str.lower().str.strip()
    if "is_error" in t.columns:
        t = t[~t["is_error"].astype(str).isin(["1", "1.0", "True", "true"])]
    t["timestamp"] = pd.to_numeric(t["timestamp"], errors="coerce")
    t = t.dropna(subset=["timestamp"])
    t["token_symbol"] = t["token_symbol"].fillna("").astype(str) if "token_symbol" in t else ""
    return t[t["from_wallet"] != t["to_wallet"]]


class FundingEvidence:
    """Per-wallet sets of funders under the definitions we sweep over."""

    def __init__(self, transfers: pd.DataFrame, first_trade: pd.Series,
                 stable_symbols=STABLE_SYMBOLS):
        self.covered: set[str] = set(transfers["queried_wallet"])
        cov = self.covered
        inbound = transfers[transfers["to_wallet"].isin(cov)
                            & (transfers["from_wallet"] != ZERO_ADDRESS)]
        inbound = inbound.drop_duplicates(["from_wallet", "to_wallet", "timestamp",
                                           "token_symbol"] + (["tx_hash"] if "tx_hash" in inbound else []))

        # direct transfers between two fetched wallets, either direction, any time
        both = transfers[transfers["from_wallet"].isin(cov) & transfers["to_wallet"].isin(cov)]
        self.direct: set[str] = {pair_key(a, b) for a, b in
                                 zip(both["from_wallet"], both["to_wallet"])}

        # funder fan-out: how many fetched wallets each sender reaches
        self.fanout: dict[str, int] = inbound.groupby("from_wallet")["to_wallet"].nunique().to_dict()

        any_time = inbound[["from_wallet", "to_wallet"]].drop_duplicates()
        self.funders_any = self._by_wallet(any_time)

        ft = inbound["to_wallet"].map(first_trade)
        is_dep = inbound["token_symbol"].isin(stable_symbols) & (inbound["timestamp"] <= ft)
        dep = inbound[is_dep][["from_wallet", "to_wallet"]].drop_duplicates()
        self.funders_deposit = self._by_wallet(dep)
        self.n_wallets_with_deposit = len(self.funders_deposit)

    @staticmethod
    def _by_wallet(df: pd.DataFrame) -> dict[str, set[str]]:
        return {w: set(g) for w, g in df.groupby("to_wallet")["from_wallet"]}

    def shared(self, a: str, b: str, kind: str, cap: float) -> set[str]:
        src = self.funders_any if kind == "any" else self.funders_deposit
        s = src.get(a, set()) & src.get(b, set())
        return {f for f in s if self.fanout.get(f, 0) <= cap}

    def has_shared(self, a, b, kind, cap) -> bool:
        return len(self.shared(a, b, kind, cap)) > 0

    def rarity_score(self, a, b, kind="deposit", cap=np.inf) -> float:
        """Adamic-Adar style: rare shared funders count more than common ones."""
        return float(sum(1.0 / np.log(1.0 + self.fanout.get(f, 1))
                         for f in self.shared(a, b, kind, cap)))


# ------------------------------------------------------------------- controls
def build_pairs(candidates: pd.DataFrame, edges: pd.DataFrame, covered: set[str],
                exclude: set[str] | None = None, controls_per_anchor: int = 1,
                seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Candidate pairs plus anchor-fixed, co-timing-matched, never-reused controls."""
    exclude = {w.lower() for w in (exclude or set())}
    usable = covered - exclude

    c = candidates[["wallet_a", "wallet_b"]].copy()
    c["wallet_a"], c["wallet_b"] = c["wallet_a"].str.lower(), c["wallet_b"].str.lower()
    c = c[c.wallet_a != c.wallet_b]
    n_cand_in = len(c)
    c = c[c.wallet_a.isin(usable) & c.wallet_b.isin(usable)].copy()
    c["pair"] = [pair_key(a, b) for a, b in zip(c.wallet_a, c.wallet_b)]
    c = c.drop_duplicates("pair")
    cand_keys = set(c["pair"])

    e = edges[["source", "target"]].copy()
    e["a"] = e[["source", "target"]].min(axis=1)
    e["b"] = e[["source", "target"]].max(axis=1)
    cnt = e.groupby(["a", "b"]).size().rename("cot").reset_index()
    cnt = cnt[cnt.a.isin(usable) & cnt.b.isin(usable)]
    cnt["pair"] = cnt["a"] + "|" + cnt["b"]
    cot = dict(zip(cnt["pair"], cnt["cot"]))

    # candidate -> partners map for each wallet, from the pool of real co-trading pairs
    partners: dict[str, list[tuple[str, float]]] = {}
    for a, b, k in zip(cnt["a"], cnt["b"], cnt["cot"]):
        if f"{a}|{b}" in cand_keys:
            continue
        partners.setdefault(a, []).append((b, np.log1p(k)))
        partners.setdefault(b, []).append((a, np.log1p(k)))

    rng = np.random.default_rng(seed)
    used: set[str] = set()
    rows = []
    n_short = 0
    for r in c.itertuples(index=False):
        target = np.log1p(cot.get(r.pair, 0))
        rows.append({"pair": r.pair, "wallet_a": r.wallet_a, "wallet_b": r.wallet_b,
                     "group": "candidate", "anchor": "", "cotiming_count": int(cot.get(r.pair, 0)),
                     "match_abs_log_diff": 0.0})
        for anchor, other in ((r.wallet_a, r.wallet_b), (r.wallet_b, r.wallet_a)):
            cands = [(p, abs(lk - target)) for p, lk in partners.get(anchor, [])
                     if p != other and pair_key(anchor, p) not in used]
            if not cands:
                n_short += 1
                continue
            cands.sort(key=lambda x: x[1])
            best = cands[0][1]
            ties = [p for p, d in cands if d <= best + 1e-12]
            take = rng.choice(len(ties), size=min(controls_per_anchor, len(ties)), replace=False)
            if controls_per_anchor > len(take):       # not enough exact ties: widen
                extra = [p for p, _ in cands if p not in {ties[i] for i in take}]
                take_p = [ties[i] for i in take] + extra[: controls_per_anchor - len(take)]
            else:
                take_p = [ties[i] for i in take]
            for p in take_p:
                k = pair_key(anchor, p)
                used.add(k)
                d = dict(cands)[p]
                rows.append({"pair": k, "wallet_a": min(anchor, p), "wallet_b": max(anchor, p),
                             "group": "control", "anchor": anchor,
                             "cotiming_count": int(cot.get(k, 0)),
                             "match_abs_log_diff": float(d)})
    out = pd.DataFrame(rows)
    info = {"candidate_pairs_input": int(n_cand_in), "candidate_pairs_used": int(len(c)),
            "candidate_pairs_dropped_no_coverage": int(n_cand_in - len(c)),
            "control_pairs": int((out.group == "control").sum()) if len(out) else 0,
            "control_slots_unfilled": int(n_short),
            "controls_reused": 0,
            "wallets_excluded": len(exclude & covered)}
    return out, info


# ------------------------------------------------------------------ statistics
def _bootstrap_diff(pairs: pd.DataFrame, x: np.ndarray, n_boot: int, seed: int
                    ) -> tuple[float, float, float]:
    """Candidate-minus-control difference with a wallet-level bootstrap 95% CI.

    Each bootstrap draw reweights wallets (multinomial); a pair's weight is the
    product of its two wallets' weights, so pairs that share wallets move
    together instead of being treated as independent observations.
    """
    wallets = sorted(set(pairs.wallet_a) | set(pairs.wallet_b))
    idx = {w: i for i, w in enumerate(wallets)}
    A = np.array([idx[w] for w in pairs.wallet_a]); B = np.array([idx[w] for w in pairs.wallet_b])
    g = (pairs.group == "candidate").to_numpy()
    x = x.astype(float)

    def diff(w):
        pw = w[A] * w[B]
        wc, wk = pw * g, pw * ~g
        if wc.sum() == 0 or wk.sum() == 0:
            return np.nan
        return (wc * x).sum() / wc.sum() - (wk * x).sum() / wk.sum()

    obs = diff(np.ones(len(wallets)))
    rng = np.random.default_rng(seed)
    p = np.full(len(wallets), 1.0 / len(wallets))
    boots = np.array([diff(rng.multinomial(len(wallets), p).astype(float)) for _ in range(n_boot)])
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return obs, np.nan, np.nan
    return obs, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def evaluate(pairs: pd.DataFrame, ev: FundingEvidence, caps=DEFAULT_CAPS,
             n_boot: int = 2000, seed: int = 7) -> pd.DataFrame:
    """One row per measure: candidate rate, control rate, gap, bootstrap CI."""
    measures: dict[str, np.ndarray] = {}
    wa, wb = pairs.wallet_a.tolist(), pairs.wallet_b.tolist()
    measures["direct_transfer (any time)"] = np.array(
        [pair_key(a, b) in ev.direct for a, b in zip(wa, wb)])
    for kind, label in (("any", "shared funder, any transfer any time"),
                        ("deposit", "shared funder, deposit-only")):
        for cap in caps:
            measures[f"{label}, funder reaches <= {cap} wallets"] = np.array(
                [ev.has_shared(a, b, kind, cap) for a, b in zip(wa, wb)])
    measures["rarity-weighted shared deposit funders (mean score)"] = np.array(
        [ev.rarity_score(a, b, "deposit") for a, b in zip(wa, wb)])

    g = (pairs.group == "candidate").to_numpy()
    rows = []
    for name, x in measures.items():
        obs, lo, hi = _bootstrap_diff(pairs, x, n_boot, seed)
        rows.append({"measure": name,
                     "candidate": float(x[g].mean()), "control": float(x[~g].mean()),
                     "difference": obs, "ci95_low": lo, "ci95_high": hi,
                     "ci_excludes_zero": bool(lo > 0 or hi < 0) if not np.isnan(lo) else False,
                     "n_candidate": int(g.sum()), "n_control": int((~g).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- driver
def run_funding_robustness(candidate_path, edge_path, trades_path, funding_path, out_dir,
                           exclude_wallets_file: str | None = None,
                           caps=DEFAULT_CAPS, n_boot: int = 2000, seed: int = 42) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(candidate_path)
    edges = pd.read_parquet(edge_path)
    trades = pd.read_parquet(trades_path)
    trades["wallet"] = trades["wallet"].astype(str).str.lower()
    first_trade = trades.groupby("wallet")["timestamp"].min()
    transfers = load_transfers(funding_path)

    exclude: set[str] = set()
    if exclude_wallets_file:
        p = Path(exclude_wallets_file)
        if p.suffix == ".csv":
            col = "queried_wallet" if "queried_wallet" in pd.read_csv(p, nrows=1).columns else None
            exclude = set(pd.read_csv(p)[col or pd.read_csv(p, nrows=1).columns[0]].astype(str).str.lower())
        else:
            exclude = {w.lower() for w in p.read_text().split()}

    ev = FundingEvidence(transfers, first_trade)
    pairs, info = build_pairs(candidates, edges, ev.covered, exclude=exclude, seed=seed)
    if (pairs.group == "control").sum() == 0 or (pairs.group == "candidate").sum() == 0:
        raise ValueError("no usable candidate/control pairs; check that the transfers file "
                         "covers the candidate wallets")
    table = evaluate(pairs, ev, caps=caps, n_boot=n_boot, seed=seed + 1)

    pairs["direct_transfer"] = [pair_key(a, b) in ev.direct for a, b in zip(pairs.wallet_a, pairs.wallet_b)]
    pairs.to_parquet(out / "funding_robustness_pairs.parquet", index=False)
    table.to_csv(out / "funding_robustness_table.csv", index=False)

    n_hubs = {c: int(sum(1 for v in ev.fanout.values() if v > c)) for c in caps}
    summary = {
        "objective": "Candidate vs control funding evidence with anchor-fixed controls and hub sweeps.",
        "inputs": {"candidates": str(candidate_path), "transfers": str(funding_path)},
        "coverage": {"wallets_with_onchain_data": len(ev.covered),
                     "wallets_with_a_pre_first_trade_stable_deposit": ev.n_wallets_with_deposit},
        "pairs": info,
        "funders_ignored_at_cap": n_hubs,
        "table": table.round(4).to_dict(orient="records"),
        "interpretation": DISCLAIMER,
        "limitations": [
            "Funding is limited to the fetched window; deposits before the window start are invisible.",
            "Wallets that hit the pagination cap are missing later history.",
            "Intervals treat wallets as the unit of resampling; they are descriptive, not causal.",
            "Many measures are reported; pick the primary one before reading results, and treat the rest as sensitivity checks.",
        ],
    }
    (out / "funding_robustness.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
