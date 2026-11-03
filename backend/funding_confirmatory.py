"""Pre-registered, coverage-matched funding test, built to be replicated.

Question: do behaviourally-flagged wallet pairs have more on-chain funding links
than comparable non-flagged pairs?

What is different from `validate-funding` / `funding-robustness`
-----------------------------------------------------------------
1. PREDEFINED. The primary measure, direction, alpha, matching covariates,
   caliper and success rule live in a YAML file whose sha256 is written to a lock
   file BEFORE the replication data is analysed. `run` refuses to start without
   a matching lock. Every non-primary measure is labelled secondary.
2. COVERAGE-MATCHED. A funding link can only be seen if the history was fetched.
   Both groups are restricted to wallets whose history is not truncated, and
   controls are matched to each candidate on
       log co-timing count, min/max log(#transfer rows) of the two wallets,
       min days of history before the first trade, max days the history ends
       before the window end,
   by nearest neighbour without replacement inside a caliper. A balance table
   (standardised mean differences) is part of the output.
3. ONE-SIDED, CLUSTER-ROBUST. Success needs the one-sided lower confidence
   bound of (candidate - control) to exceed `min_effect`, from a wallet-level
   bootstrap (pairs share wallets), in EACH replication market and pooled.
4. POWER IS REPORTED. `power` shows what effect size a given number of pairs can
   detect, so an "inconclusive" is distinguishable from a "no effect".
5. REPLICATION ROLES. `--role discovery|replication` is recorded; `combine`
   applies the rule across markets.

Commands
--------
  lock     hash + timestamp the pre-registration           (do this first)
  run      one market: matching, balance, test, verdict
  combine  inverse-variance pooled estimate across runs and the overall verdict
  power    minimum detectable effect for given sample sizes

A funding link is evidence of an observed transfer, never proof of common
ownership. Exchanges/bridges/routers create links between unrelated wallets;
pre-register your hub cap and keep it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from polymap.analysis.funding_robustness import FundingEvidence, load_transfers, pair_key
from polymap.config import load_config

COVARIATES = ["log_cot", "min_log_rows", "max_log_rows", "min_pre_trade_days", "max_end_gap_days"]


# ------------------------------------------------------------------- utils
def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock_path(prereg) -> Path:
    return Path(str(prereg) + ".lock.json")


def read_prereg(prereg) -> dict:
    prereg = Path(prereg)
    lp = lock_path(prereg)
    if not lp.exists():
        sys.exit(f"no lock for {prereg}. Run `lock` first, before analysing replication data.")
    lock = json.loads(lp.read_text())
    if lock["sha256"] != sha256_file(prereg):
        sys.exit("PRE-REGISTRATION VIOLATION: the prereg file changed after it was locked.")
    return yaml.safe_load(prereg.read_text())


# ---------------------------------------------------------------- coverage
def wallet_coverage(transfers: pd.DataFrame, trades: pd.DataFrame, truncation_rows: int,
                    truncated_file: str | None, recovery_report: str | None = None) -> pd.DataFrame:
    """Per-wallet coverage covariates and a truncation flag.

    Truncation source, in order of precedence:
      * `truncated_file` given (output of find_truncated_wallets.py, run AFTER
        recover_erc20.py): a wallet is truncated iff that file flags it by the
        row/end-gap heuristic, or the recovery report still lists a capped block
        range for it. The raw row-count threshold is NOT applied, because a
        recovered wallet legitimately has > 10,000 complete rows.
      * otherwise: rows >= truncation_rows (pre-recovery fallback).
    """
    g = transfers.groupby("queried_wallet")["timestamp"].agg(rows="size", first_ts="min", last_ts="max")
    first_trade = trades.groupby("wallet")["timestamp"].min()
    window_end = trades["timestamp"].max()
    g["first_trade"] = first_trade.reindex(g.index)
    g["pre_trade_days"] = ((g["first_trade"] - g["first_ts"]) / 86400).clip(-30, 180)
    g["end_gap_days"] = ((window_end - g["last_ts"]) / 86400).clip(0, 365)
    g["log_rows"] = np.log1p(g["rows"])
    if truncated_file:
        t = pd.read_csv(truncated_file)
        t["queried_wallet"] = t["queried_wallet"].astype(str).str.lower()
        if "flagged_by_row_heuristic" in t.columns:
            t = t[t["flagged_by_row_heuristic"].astype(str).str.lower().isin(["true", "1"])]
        flagged = set(t["queried_wallet"])
        if recovery_report and Path(recovery_report).exists():
            rep = json.loads(Path(recovery_report).read_text())
            flagged |= {r["address"].lower() for r in rep.get("remaining_truncated_ranges", [])}
        g["truncated"] = g.index.isin(flagged)
    else:
        g["truncated"] = g["rows"] >= truncation_rows
    return g.dropna(subset=["first_trade"])


# ---------------------------------------------------------------- matching
def pair_covariates(df: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    a = cov.reindex(df["wallet_a"])
    b = cov.reindex(df["wallet_b"])
    out = df.copy()
    out["min_log_rows"] = np.minimum(a["log_rows"].to_numpy(), b["log_rows"].to_numpy())
    out["max_log_rows"] = np.maximum(a["log_rows"].to_numpy(), b["log_rows"].to_numpy())
    out["min_pre_trade_days"] = np.minimum(a["pre_trade_days"].to_numpy(), b["pre_trade_days"].to_numpy())
    out["max_end_gap_days"] = np.maximum(a["end_gap_days"].to_numpy(), b["end_gap_days"].to_numpy())
    return out


def build_matched(candidates: pd.DataFrame, edges: pd.DataFrame, cov: pd.DataFrame,
                  excluded: set[str], m: int, caliper: float, seed: int):
    usable = set(cov.index[~cov["truncated"]]) - excluded
    c = candidates[["wallet_a", "wallet_b"]].astype(str).apply(lambda s: s.str.lower())
    n_in = len(c)
    c = c[(c.wallet_a != c.wallet_b) & c.wallet_a.isin(usable) & c.wallet_b.isin(usable)].copy()
    c["pair"] = [pair_key(x, y) for x, y in zip(c.wallet_a, c.wallet_b)]
    c = c.drop_duplicates("pair").reset_index(drop=True)

    e = edges[["source", "target"]].astype(str)
    cnt = (pd.DataFrame({"wallet_a": e.min(axis=1), "wallet_b": e.max(axis=1)})
           .groupby(["wallet_a", "wallet_b"]).size().rename("cot").reset_index())
    cnt = cnt[cnt.wallet_a.isin(usable) & cnt.wallet_b.isin(usable)]
    cnt["pair"] = cnt.wallet_a + "|" + cnt.wallet_b
    cot = dict(zip(cnt["pair"], cnt["cot"]))
    c["cot"] = c["pair"].map(cot).fillna(0).astype(int)
    pool = cnt[~cnt["pair"].isin(set(c["pair"]))].reset_index(drop=True)
    for df in (c, pool):
        df["log_cot"] = np.log1p(df["cot"])
    c, pool = pair_covariates(c, cov), pair_covariates(pool, cov)
    c = c.dropna(subset=COVARIATES).reset_index(drop=True)
    pool = pool.dropna(subset=COVARIATES).reset_index(drop=True)
    if len(c) == 0 or len(pool) == 0:
        sys.exit("no usable candidate or control pairs after coverage restriction")

    mu, sd = pool[COVARIATES].mean(), pool[COVARIATES].std().replace(0, 1.0).fillna(1.0)
    zc = ((c[COVARIATES] - mu) / sd).to_numpy()
    zp = ((pool[COVARIATES] - mu) / sd).to_numpy()
    tree = cKDTree(zp)
    rng = np.random.default_rng(seed)
    used: set[int] = set()
    rows, short, dropped = [], 0, 0
    k = min(300, len(pool))
    for i in rng.permutation(len(c)):
        dist, ind = tree.query(zc[i], k=k)
        dist, ind = np.atleast_1d(dist), np.atleast_1d(ind)
        take = [j for dj, j in zip(dist, ind) if dj <= caliper and j not in used][:m]
        if not take:
            dropped += 1
            continue
        short += int(len(take) < m)
        rows.append({**c.iloc[i][["pair", "wallet_a", "wallet_b", *COVARIATES]].to_dict(),
                     "group": "candidate", "set_id": int(i), "match_dist": 0.0})
        for j in take:
            used.add(int(j))
            r = pool.iloc[int(j)]
            rows.append({**r[["pair", "wallet_a", "wallet_b", *COVARIATES]].to_dict(),
                         "group": "control", "set_id": int(i), "match_dist": float(dist[list(ind).index(j)])})
    pairs = pd.DataFrame(rows)
    info = {"candidate_pairs_input": int(n_in), "candidate_pairs_usable": int(len(c)),
            "candidate_pairs_dropped_no_match_in_caliper": int(dropped),
            "candidate_pairs_with_fewer_than_m_controls": int(short),
            "candidates_in_analysis": int((pairs.group == "candidate").sum()),
            "controls_in_analysis": int((pairs.group == "control").sum()),
            "wallets_usable": int(len(usable))}
    return pairs, info


def balance_table(pairs: pd.DataFrame) -> pd.DataFrame:
    g = pairs.group == "candidate"
    rows = []
    for col in COVARIATES:
        x1, x0 = pairs.loc[g, col], pairs.loc[~g, col]
        pooled = np.sqrt((x1.var() + x0.var()) / 2) or 1.0
        rows.append({"covariate": col, "candidate_mean": x1.mean(), "control_mean": x0.mean(),
                     "smd": (x1.mean() - x0.mean()) / pooled})
    t = pd.DataFrame(rows)
    t["balanced_abs_smd_lt_0.1"] = t["smd"].abs() < 0.1
    return t


# -------------------------------------------------------------- statistics
def cluster_bootstrap(pairs: pd.DataFrame, x: np.ndarray, n_boot: int, seed: int):
    """Candidate-minus-control rate difference; resample WALLETS, pair weight = product."""
    wallets = sorted(set(pairs.wallet_a) | set(pairs.wallet_b))
    idx = {w: i for i, w in enumerate(wallets)}
    A = np.array([idx[w] for w in pairs.wallet_a])
    B = np.array([idx[w] for w in pairs.wallet_b])
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
    return obs, boots[~np.isnan(boots)]


def measure_values(pairs: pd.DataFrame, ev: FundingEvidence, measure: str, cap: int) -> np.ndarray:
    wa, wb = pairs.wallet_a.tolist(), pairs.wallet_b.tolist()
    if measure == "direct_transfer":
        return np.array([pair_key(a, b) in ev.direct for a, b in zip(wa, wb)])
    if measure == "shared_deposit_funder":
        return np.array([ev.has_shared(a, b, "deposit", cap) for a, b in zip(wa, wb)])
    if measure == "shared_any_funder":
        return np.array([ev.has_shared(a, b, "any", cap) for a, b in zip(wa, wb)])
    raise ValueError(f"unknown measure {measure}")


# ---------------------------------------------------------------- commands
def cmd_lock(a) -> int:
    lp = lock_path(a.prereg)
    if lp.exists():
        sys.exit(f"{lp} already exists; a pre-registration is not re-locked. Create a new version.")
    lp.write_text(json.dumps({"prereg": str(a.prereg), "sha256": sha256_file(a.prereg),
                              "locked_at_utc": datetime.now(timezone.utc).isoformat(),
                              "note": a.note or ""}, indent=2))
    print(f"locked {a.prereg} -> {lp}\nCommit both files before touching replication data.")
    return 0


def cmd_run(a) -> int:
    pr = read_prereg(a.prereg)
    cfg = load_config(run_name=a.run_name)
    proc, art = cfg.processed_dir, cfg.artifacts_dir
    art.mkdir(parents=True, exist_ok=True)
    cand_f = a.candidates_file or proc / "candidate_pairs.parquet"
    candidates = pd.read_parquet(cand_f)
    edges = pd.read_parquet(a.edges_file or proc / "temporal_edges.parquet")
    trades = pd.read_parquet(a.trades_file or proc / "trades.parquet")
    trades["wallet"] = trades["wallet"].astype(str).str.lower()
    transfers = load_transfers(a.funding_file)

    rec = a.recovery_report or str(Path(a.funding_file).parent / "erc20_recovery_report.json")
    cov = wallet_coverage(transfers, trades, int(pr["truncation_rows"]), a.truncated_file, rec)
    excluded = set()
    if a.exclude_file:
        excluded = {w.lower() for w in Path(a.exclude_file).read_text().split()}
    pairs, info = build_matched(candidates, edges, cov, excluded, int(pr["controls_per_candidate"]),
                                float(pr["caliper_sd"]), int(pr["seed"]))
    first_trade = trades.groupby("wallet")["timestamp"].min()
    ev = FundingEvidence(transfers, first_trade)
    cap = int(pr["shared_funder_cap"])

    primary = pr["primary_measure"]
    x = measure_values(pairs, ev, primary, cap)
    obs, boots = cluster_bootstrap(pairs, x, int(pr["n_boot"]), int(pr["seed"]) + 1)
    alpha = float(pr["alpha"])
    lb = float(np.percentile(boots, 100 * alpha))                   # one-sided
    lo2, hi2 = np.percentile(boots, [2.5, 97.5])
    g = (pairs.group == "candidate").to_numpy()
    bal = balance_table(pairs)
    n_c = info["candidates_in_analysis"]

    if n_c < int(pr["min_candidate_pairs"]):
        verdict = "inconclusive_too_few_candidate_pairs"
    elif lb > float(pr["min_effect"]):
        verdict = "supported"
    else:
        verdict = "not_supported"

    secondary = {}
    for meas in ("direct_transfer", "shared_deposit_funder", "shared_any_funder"):
        if meas == primary:
            continue
        xs = measure_values(pairs, ev, meas, cap)
        o, b = cluster_bootstrap(pairs, xs, int(pr["n_boot"]), int(pr["seed"]) + 2)
        secondary[meas] = {"candidate_rate": float(xs[g].mean()), "control_rate": float(xs[~g].mean()),
                           "difference": float(o), "ci95": [float(np.percentile(b, 2.5)),
                                                            float(np.percentile(b, 97.5))],
                           "label": "SECONDARY - not part of the decision rule"}

    result = {"prereg": str(a.prereg), "prereg_sha256": sha256_file(a.prereg), "role": a.role,
              "run_name": a.run_name, "primary_measure": primary, "shared_funder_cap": cap,
              "candidate_rate": float(x[g].mean()), "control_rate": float(x[~g].mean()),
              "difference": float(obs), "se": float(boots.std(ddof=1)),
              "one_sided_lower_bound": lb, "alpha_one_sided": alpha,
              "ci95_two_sided": [float(lo2), float(hi2)], "min_effect": float(pr["min_effect"]),
              "verdict": verdict, "matching": info,
              "balance_ok": bool(bal["balanced_abs_smd_lt_0.1"].all()),
              "balance": bal.round(4).to_dict(orient="records"),
              "mean_match_distance": float(pairs.loc[~g, "match_dist"].mean()),
              "coverage": {"wallets_with_history": int(len(cov)),
                           "wallets_truncated": int(cov["truncated"].sum())},
              "secondary": secondary,
              "inputs": {"funding_sha256": sha256_file(a.funding_file),
                         "candidates_sha256": sha256_file(cand_f)},
              "wallets": sorted(set(pairs.wallet_a) | set(pairs.wallet_b)),
              "limitations": ["Funding link = observed transfer, not ownership.",
                              "Matching adjusts for observed coverage covariates only.",
                              "Bootstrap treats wallets as the resampling unit; markets sharing wallets "
                              "are not independent (see combine)."]}
    out = art / f"funding_confirmatory_{a.role}.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    pairs.assign(outcome=x).to_parquet(art / f"funding_confirmatory_{a.role}_pairs.parquet", index=False)
    print(f"{a.role}/{a.run_name}: candidates {n_c}, controls {info['controls_in_analysis']}")
    print(f"{primary}: candidate {x[g].mean():.3f} vs control {x[~g].mean():.3f} "
          f"diff {obs:+.3f}, one-sided {int(100 * (1 - alpha))}% lower bound {lb:+.3f}")
    print(f"balance ok: {result['balance_ok']}  ->  VERDICT: {verdict}\nwrote {out}")
    return 0


def cmd_combine(a) -> int:
    pr = read_prereg(a.prereg)
    runs = [json.loads(Path(f).read_text()) for f in a.results]
    for r in runs:
        if r["prereg_sha256"] != sha256_file(a.prereg):
            sys.exit(f"{r['run_name']} was produced under a different pre-registration")
    reps = [r for r in runs if r["role"] == "replication"]
    if not reps:
        sys.exit("combine needs at least one replication run")
    w = np.array([1 / r["se"] ** 2 for r in reps])
    est = np.array([r["difference"] for r in reps])
    pooled = float((w * est).sum() / w.sum())
    se = float(np.sqrt(1 / w.sum()))
    z = {0.05: 1.645, 0.025: 1.96, 0.01: 2.326}.get(float(pr["alpha"]), 1.645)
    lb = pooled - z * se
    sets = [set(r["wallets"]) for r in reps]
    shared = len(set.intersection(*sets)) if len(sets) > 1 else 0
    each_ok = all(r["verdict"] == "supported" for r in reps)
    overall = "supported" if (each_ok and lb > float(pr["min_effect"])) else (
        "inconclusive" if any(r["verdict"].startswith("inconclusive") for r in reps) else "not_supported")
    out = {"rule": "supported iff every replication market is 'supported' AND the pooled one-sided "
                   "lower bound exceeds min_effect",
           "pooled_difference": pooled, "pooled_se": se, "pooled_one_sided_lower_bound": lb,
           "per_market": [{k: r[k] for k in ("run_name", "role", "difference", "se",
                                             "one_sided_lower_bound", "verdict", "balance_ok")}
                          for r in runs],
           "wallets_in_all_replication_markets": shared,
           "independence_caveat": "Pooling assumes markets are independent; shared wallets violate this.",
           "overall_verdict": overall}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


def cmd_power(a) -> int:
    from math import sqrt
    from statistics import NormalDist
    nd = NormalDist()
    za, zb = nd.inv_cdf(1 - a.alpha), nd.inv_cdf(a.power)
    p0 = a.control_rate
    n1, n0, deff = a.n_candidates, a.n_candidates * a.controls_per_candidate, a.design_effect
    print(f"control rate {p0:.3f}, {n1} candidates, {n0} controls, design effect {deff}")
    for eff in a.effects:
        p1 = p0 + eff
        se = sqrt((p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0) * deff)
        pw = 1 - nd.cdf(za - eff / se)
        print(f"  true effect {eff:+.3f}: power ~ {pw:.2f}")
    se0 = sqrt((p0 * (1 - p0) / n1 + p0 * (1 - p0) / n0) * deff)
    print(f"minimum detectable effect at {a.power:.0%} power ~ {(za + zb) * se0:+.3f} "
          f"(normal approximation; the design effect is an ASSUMPTION - estimate it from your bootstrap)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("lock")
    l.add_argument("--prereg", required=True)
    l.add_argument("--note")
    r = sub.add_parser("run")
    r.add_argument("--prereg", required=True)
    r.add_argument("--run-name", required=True)
    r.add_argument("--funding-file", required=True)
    r.add_argument("--role", choices=["discovery", "replication"], required=True)
    r.add_argument("--candidates-file")
    r.add_argument("--edges-file")
    r.add_argument("--trades-file")
    r.add_argument("--truncated-file", help="csv with queried_wallet, flagged_by_row_heuristic")
    r.add_argument("--exclude-file", help="txt of wallets to drop (e.g. known service addresses)")
    r.add_argument("--recovery-report", help="erc20_recovery_report.json (default: next to --funding-file)")
    c = sub.add_parser("combine")
    c.add_argument("--prereg", required=True)
    c.add_argument("--results", nargs="+", required=True)
    c.add_argument("--out", default="funding_confirmatory_combined.json")
    w = sub.add_parser("power")
    w.add_argument("--n-candidates", type=int, required=True)
    w.add_argument("--controls-per-candidate", type=int, default=2)
    w.add_argument("--control-rate", type=float, required=True)
    w.add_argument("--effects", type=float, nargs="+", default=[0.03, 0.05, 0.07, 0.10])
    w.add_argument("--design-effect", type=float, default=1.5)
    w.add_argument("--alpha", type=float, default=0.05)
    w.add_argument("--power", type=float, default=0.8)
    a = p.parse_args(argv)
    return {"lock": cmd_lock, "run": cmd_run, "combine": cmd_combine, "power": cmd_power}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
