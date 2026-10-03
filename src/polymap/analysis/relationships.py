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
    lo = np.where(a.values <= b.values, a.values, b.values)
    hi = np.where(a.values <= b.values, b.values, a.values)
    return pd.Series([f"{x}|{y}" for x, y in zip(lo, hi)], index=a.index)


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
        span_seconds=("timestamp", lambda s: int(s.max() - s.min())),
        pct_same_side=("same_side", "mean"),
        pct_same_outcome=("same_outcome", "mean"),
        pct_same_behavior=("same_behavior", "mean"),
        median_time_diff=("time_diff", "median"),
        median_price_diff=("price_diff", "median"),
        median_size_ratio=("size_ratio", "median"),
    )
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
    return p.sort_values("mean_margin", ascending=False).reset_index(drop=True)


def run_screening(events: pd.DataFrame, edges: pd.DataFrame,
                  cfg: ScreeningConfig | None = None) -> pd.DataFrame:
    cfg = cfg or ScreeningConfig()
    pairs = aggregate_pairs(events, cfg.min_events)
    pairs = characterise_pairs(pairs, edges)
    return screen_pairs(pairs, cfg)
