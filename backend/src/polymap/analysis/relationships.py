"""Turn per-event model scores into ranked, screened wallet pairs.

This is not another model. It aggregates margins and applies two binary
criteria, and the vocabulary is constrained on purpose.

**Read the caveat in `screen_pairs` before using the output anywhere.**
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import ScreeningConfig

log = logging.getLogger(__name__)


def _pair_key(a: pd.Series, b: pd.Series) -> pd.Series:
    """Undirected pair key, so A->B and B->A aggregate together."""
    lo = pd.Series(np.where(a.values <= b.values, a.values, b.values), index=a.index)
    hi = pd.Series(np.where(a.values <= b.values, b.values, a.values), index=a.index)
    return lo.astype(str) + "|" + hi.astype(str)


def aggregate_pairs(events: pd.DataFrame, min_events: int = 3) -> pd.DataFrame:
    """Collapse scored events into one row per wallet pair."""
    e = events.copy()
    e["pair"] = _pair_key(e["source"], e["target"])
    g = e.groupby("pair", as_index=False).agg(
        n_events=("margin", "size"),
        mean_margin=("margin", "mean"),
        std_margin=("margin", "std"),
        mean_positive_score=("positive_score", "mean"),
        first_event=("timestamp", "min"),
        last_event=("timestamp", "max"),
    )
    g["std_margin"] = g["std_margin"].fillna(0.0)
    g[["wallet_a", "wallet_b"]] = g["pair"].str.split("|", expand=True)
    g = g[g["n_events"] >= min_events].copy()
    return g.sort_values("mean_margin", ascending=False).reset_index(drop=True)


def characterise_pairs(pairs: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Attach observed interaction statistics to each pair.

    Vectorised over all pairs at once. The notebook version filtered the full
    edge frame separately for every pair, which is O(pairs * edges) and the
    second-worst bottleneck in the original code after edge construction.
    """
    e = edges.copy()
    e["pair"] = _pair_key(e["source"], e["target"])
    stats = e.groupby("pair", as_index=False).agg(
        real_events=("timestamp", "size"),
        distinct_timestamps=("timestamp", "nunique"),
        ts_min=("timestamp", "min"),
        ts_max=("timestamp", "max"),
        pct_same_side=("same_side", "mean"),
        pct_same_outcome=("same_outcome", "mean"),
        pct_same_behavior=("same_behavior", "mean"),
        median_time_diff=("time_diff", "median"),
        median_price_diff=("price_diff", "median"),
        median_size_ratio=("size_ratio", "median"),
    )
    stats["span_seconds"] = (stats["ts_max"] - stats["ts_min"]).astype("int64")
    stats = stats.drop(columns=["ts_min", "ts_max"])
    stats["span_days"] = stats["span_seconds"] / 86400.0
    return pairs.merge(stats, on="pair", how="left")


def screen_pairs(pairs: pd.DataFrame, cfg: ScreeningConfig | None = None) -> pd.DataFrame:
    """Apply the persistence and tightness criteria.

    CAVEAT, and it belongs in every writeup that uses this output:

    Tightness is measured *inside a window we imposed*. Edge construction caps
    interactions at `window_seconds`, so the time-difference distribution is
    already truncated before this threshold is applied. These are screening
    heuristics for directing human attention, not measurements of synchrony and
    not evidence of coordination, shared control, or shared identity.

    Two wallets reacting independently to the same news produce exactly the
    same pattern as two wallets under one operator. Nothing here separates them.
    """
    cfg = cfg or ScreeningConfig()
    p = pairs.copy()

    p["persistent"] = (
        (p["span_days"] > cfg.persistent_min_span_days)
        & (p["distinct_timestamps"] >= cfg.persistent_min_timestamps)
    ).fillna(False)

    p["tight"] = (
        (p["median_time_diff"] <= cfg.tight_max_median_time_diff)
        & (p["median_price_diff"] <= cfg.tight_max_median_price_diff)
    ).fillna(False)

    p["category"] = np.select(
        [p["persistent"] & p["tight"], p["persistent"] & ~p["tight"],
         ~p["persistent"] & p["tight"]],
        ["persistent_tight", "persistent_loose", "one_off_tight"],
        default="one_off_loose")

    counts = p["category"].value_counts().to_dict()
    log.info("screening categories: %s", counts)
    log.info("NOTE: these are candidate relationships, not detected coordination")
    sort_col = "mean_margin" if "mean_margin" in p.columns else "real_events"
    return p.sort_values(sort_col, ascending=False).reset_index(drop=True)


def run_screening(events: pd.DataFrame, edges: pd.DataFrame,
                  cfg: ScreeningConfig | None = None) -> pd.DataFrame:
    cfg = cfg or ScreeningConfig()
    pairs = aggregate_pairs(events, cfg.min_events)
    pairs = characterise_pairs(pairs, edges)
    return screen_pairs(pairs, cfg)



# --------------------------------------------------------------------------
# Model-free candidate pre-screen
# --------------------------------------------------------------------------
CATEGORY_PRIORITY = ["persistent_tight", "one_off_tight",
                     "persistent_loose", "one_off_loose"]
DEFAULT_CANDIDATE_CATEGORIES = ("persistent_tight", "one_off_tight")


def prescreen_pairs(edges: pd.DataFrame, cfg: ScreeningConfig | None = None) -> pd.DataFrame:
    """Screen wallet pairs straight from the temporal edges, no model involved.

    The persistence and tightness criteria in `screen_pairs` only ever read
    edge statistics (span, distinct timestamps, median time/price gap). The
    model contributes the `mean_margin` ranking and nothing else, so the
    candidate set can be found in seconds with pandas, before any training and
    before any on-chain fetching.

    Same caveat as `screen_pairs`: these are screening heuristics for directing
    attention, not evidence of coordination or shared control.
    """
    cfg = cfg or ScreeningConfig()
    e = edges[["source", "target"]].copy()
    pairs = pd.DataFrame({"pair": _pair_key(e["source"], e["target"])})
    counts = pairs["pair"].value_counts()
    keep = counts[counts >= cfg.min_events].index
    pairs = pd.DataFrame({"pair": keep})
    if pairs.empty:
        return pairs.assign(wallet_a=[], wallet_b=[], category=[])
    pairs[["wallet_a", "wallet_b"]] = pairs["pair"].str.split("|", expand=True)
    pairs["n_events"] = counts.loc[keep].to_numpy()
    sub = edges[_pair_key(edges["source"], edges["target"]).isin(set(keep)).to_numpy()]
    out = characterise_pairs(pairs, sub)
    return screen_pairs(out, cfg)


def select_candidates(screened: pd.DataFrame,
                      categories: tuple[str, ...] = DEFAULT_CANDIDATE_CATEGORIES,
                      max_pairs: int = 0) -> pd.DataFrame:
    """Pick the pairs worth spending on-chain API calls on.

    Ordered by category priority, then by number of observed interactions.
    `max_pairs` > 0 keeps only the top N pairs (so the wallet list stays short).
    """
    c = screened[screened["category"].isin(categories)].copy()
    rank = {name: i for i, name in enumerate(CATEGORY_PRIORITY)}
    c["_rank"] = c["category"].map(rank)
    c = c.sort_values(["_rank", "real_events"], ascending=[True, False]).drop(columns="_rank")
    if max_pairs > 0:
        c = c.head(max_pairs)
    return c.reset_index(drop=True)


def candidate_wallets(candidates: pd.DataFrame) -> list[str]:
    return sorted(set(candidates["wallet_a"]) | set(candidates["wallet_b"]))
