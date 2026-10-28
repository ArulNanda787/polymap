"""Null models for the tightness screen.

A null model is fake data in which the thing being looked for (wallet pairs
that trade together) is deliberately absent. Running the same screen on many
fake datasets shows how many "tight" pairs appear by chance, which is the only
way to say whether the observed count is large.

stratified_side (PRIMARY, strictest) Permute wallet labels only among trades
           in the same time bin AND with the same side/outcome. Market-wide
           bursts, each wallet's coarse activity profile and its directional
           bias all survive; only fine-grained (sub-bin) pairing is
           randomised. Beating this is the minimum bar for "pair-specific".

stratified Same, but ignores side/outcome. Keeps bursts and activity, but a
           wallet that always buys YES is detached from that habit, so this
           null is too easy on a directional criterion. Report it next to
           stratified_side, never alone.

wallet     Permute wallet labels across the whole market. Destroys each
           wallet's burstiness AND the time profile, so it is biased low.
           It can rule things out ("nothing here") but not confirm.

shift, timestamp   (SANITY BOUNDS ONLY) Beating these proves nothing.

All nulls are applied within each market separately.

Scope note: the null runs the screen on edges alone. It does not retrain the
model, so its observed count differs from candidate_relationships.parquet.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..features.edges import build_edges_multi_market

log = logging.getLogger(__name__)

MODES = ("wallet", "stratified", "stratified_side", "shift", "timestamp")
DEFAULT_MODES = ("stratified", "stratified_side")
DEFAULT_BIN_SECONDS = 3600


def count_tight_pairs(edges: pd.DataFrame, min_events: int,
                      max_time_diff: float, max_price_diff: float,
                      min_same_behavior: float = 0.0) -> dict:
    """Edge-only version of the tight criterion.

    `n_tight_timing_only` applies time and price gaps alone (the old rule).
    `n_tight` additionally requires the pair to trade the same side AND outcome
    in at least `min_same_behavior` of their edges, so opposite-side matches
    (a buyer and a seller) no longer count. With min_same_behavior=0 the two
    numbers are identical.
    """
    if edges.empty:
        return {"n_pairs": 0, "n_tight": 0, "n_tight_timing_only": 0}
    codes, _ = pd.factorize(pd.concat([edges["source"], edges["target"]]))
    n = len(edges)
    a, b = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
    base = int(codes.max()) + 1
    key = np.minimum(a, b) * base + np.maximum(a, b)
    if "market_id" in edges.columns:      # same wallet pair in two markets is two pairs
        key = pd.factorize(edges["market_id"].astype(str) + "|" + pd.Series(key).astype(str))[0]

    g = (pd.DataFrame({"key": key,
                       "td": edges["time_diff"].to_numpy(),
                       "pdf": edges["price_diff"].to_numpy(),
                       "sb": edges["same_behavior"].to_numpy() if "same_behavior" in edges.columns else np.ones(len(edges))})
         .groupby("key").agg(n=("td", "size"), td=("td", "median"),
                             pdf=("pdf", "median"), sb=("sb", "mean")))
    g = g[g["n"] >= min_events]
    timing = (g["td"] <= max_time_diff) & (g["pdf"] <= max_price_diff)
    tight = timing & (g["sb"] >= min_same_behavior)
    return {"n_pairs": int(len(g)), "n_tight": int(tight.sum()),
            "n_tight_timing_only": int(timing.sum())}


def _null_market(t: pd.DataFrame, mode: str, rng: np.random.Generator,
                 bin_seconds: int = DEFAULT_BIN_SECONDS) -> pd.DataFrame:
    t = t.copy()
    if mode == "wallet":
        t["wallet"] = rng.permutation(t["wallet"].to_numpy())
    elif mode in ("stratified", "stratified_side"):
        # Permute wallet labels only *within* coarse time bins (and, for
        # "stratified_side", within side/outcome too). Each bin keeps
        # its trades, timing, prices and the same multiset of wallets, so
        # market-wide intensity and each wallet's coarse activity profile
        # survive. Only who-traded-with-whom inside a bin is randomised.
        wallets = t["wallet"].to_numpy()
        ts = t["timestamp"].to_numpy(np.int64)
        bins = (ts - ts.min()) // max(int(bin_seconds), 1)
        if mode == "stratified_side":
            # Also hold each trade's side+outcome fixed: a wallet only swaps with
            # trades in the same direction, so persistent directional bias
            # (always buying YES) cannot masquerade as coordination.
            grp = pd.factorize(t["side"].astype(str) + "|" + t["outcome"].astype(str))[0]
            bins = bins * (int(grp.max()) + 1) + grp
        by_bin = np.argsort(bins, kind="stable")                 # indices grouped by bin
        shuffled = np.lexsort((rng.random(len(t)), bins))        # same groups, random order
        new = np.empty_like(wallets)
        new[by_bin] = wallets[shuffled]
        t["wallet"] = new
    elif mode == "timestamp":
        t["timestamp"] = rng.permutation(t["timestamp"].to_numpy())
    elif mode == "shift":
        ts = t["timestamp"].to_numpy(np.int64)
        lo = int(ts.min())
        span = int(ts.max()) - lo + 1
        wallets = t["wallet"].to_numpy()
        uniq = np.unique(wallets)
        offsets = dict(zip(uniq, rng.integers(0, span, size=len(uniq))))
        off = np.fromiter((offsets[w] for w in wallets), dtype=np.int64, count=len(wallets))
        t["timestamp"] = lo + (ts - lo + off) % span
    else:
        raise ValueError(f"unknown null mode '{mode}' (choose from {MODES})")
    return t


def make_null_trades(trades: pd.DataFrame, mode: str, rng: np.random.Generator,
                     bin_seconds: int = DEFAULT_BIN_SECONDS) -> pd.DataFrame:
    parts = [_null_market(g, mode, rng, bin_seconds)
             for _, g in trades.groupby("market_id", sort=False)]
    return pd.concat(parts, ignore_index=True)


def _edges(cfg: Config, trades: pd.DataFrame) -> pd.DataFrame:
    return build_edges_multi_market(
        trades,
        window_seconds=cfg.graph.window_seconds,
        max_pairs_per_trade=cfg.graph.max_pairs_per_trade,
        drop_crowd_timestamps=cfg.graph.drop_crowd_timestamps,
        chunk_size=cfg.graph.chunk_size,
    )


def _summarise(obs: int, draws: np.ndarray) -> dict:
    std = float(draws.std(ddof=1)) if len(draws) > 1 else float("nan")
    return {
        "null_mean": float(draws.mean()),
        "null_std": std,
        "null_min": int(draws.min()),
        "null_max": int(draws.max()),
        "z": float((obs - draws.mean()) / std) if std and std > 0 else None,
        # add-one estimator: never reports exactly zero; floor is 1/(n_null+1)
        "empirical_p": float((1 + (draws >= obs).sum()) / (1 + len(draws))),
    }


def run_null_model(cfg: Config, trades: pd.DataFrame, n_null: int = 20,
                   modes: tuple[str, ...] = DEFAULT_MODES,
                   bin_seconds: int = DEFAULT_BIN_SECONDS) -> dict:
    s = cfg.screening
    min_sb = float(getattr(s, "tight_min_same_behavior", 0.8))
    args = (s.min_events, s.tight_max_median_time_diff, s.tight_max_median_price_diff, min_sb)

    observed = count_tight_pairs(_edges(cfg, trades), *args)
    log.info("observed (edge-only screen, same_behavior>=%.2f): %s", min_sb, observed)

    out = {"window_seconds": cfg.graph.window_seconds, "n_null": n_null,
           "min_same_behavior": min_sb, "bin_seconds": bin_seconds,
           "observed": observed, "nulls": {}}
    for mode in modes:
        rng = np.random.default_rng(cfg.model.seed)
        runs = [count_tight_pairs(_edges(cfg, make_null_trades(trades, mode, rng, bin_seconds)), *args)
                for _ in range(n_null)]
        tight = np.array([r["n_tight"] for r in runs])
        timing = np.array([r["n_tight_timing_only"] for r in runs])
        pairs = np.array([r["n_pairs"] for r in runs])
        res = {
            "null_tight_mean": float(tight.mean()),
            "null_tight_std": float(tight.std(ddof=1)) if n_null > 1 else float("nan"),
            "null_tight_min": int(tight.min()),
            "null_tight_max": int(tight.max()),
            "null_pairs_mean": float(pairs.mean()),
            **{k: v for k, v in _summarise(observed["n_tight"], tight).items()
               if k in ("z", "empirical_p")},
            "timing_only": _summarise(observed["n_tight_timing_only"], timing),
        }
        out["nulls"][mode] = res
        log.info("[null:%s] directional: observed %d vs null %.1f +/- %.1f (min %d, max %d), z=%s p=%.4f | "
                 "timing-only: observed %d vs null %.1f, p=%.4f | null pairs %.0f vs observed %d",
                 mode, observed["n_tight"], res["null_tight_mean"], res["null_tight_std"],
                 res["null_tight_min"], res["null_tight_max"],
                 None if res["z"] is None else round(res["z"], 1), res["empirical_p"],
                 observed["n_tight_timing_only"], res["timing_only"]["null_mean"],
                 res["timing_only"]["empirical_p"], res["null_pairs_mean"], observed["n_pairs"])
    return out