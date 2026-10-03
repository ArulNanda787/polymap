"""Temporal interaction edges, built without Python loops.

The original implementation used nested `for` loops with `.iloc` on every
access, which costs roughly O(n * k) *Python-level* operations where k is the
average number of trades inside the window. At 10,000 trades that is slow but
survivable; at 100,000 across several markets it is not, and `.iloc` on a
DataFrame row is about a thousand times more expensive than indexing a numpy
array.

This version does the same work with `searchsorted` plus array arithmetic.
Output is identical; see `tests/test_edges.py`, which asserts equality against
a literal transcription of the original loop.

The window join is inherently quadratic in the size of a burst: if 500 trades
land in the same second, that second alone produces ~125,000 pairs. Two guards
handle this. `max_pairs_per_trade` caps the fan-out of any single trade, and
`drop_crowd_timestamps` removes moments where the whole market reacts at once,
which are the moments that produce co-timing without any relationship.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..data.schema import EDGE_COLUMNS, validate_edges

log = logging.getLogger(__name__)


def _window_pairs(ts: np.ndarray, window: int, max_fanout: int,
                  lo_row: int = 0, hi_row: int | None = None
                  ) -> tuple[np.ndarray, np.ndarray, int]:
    """Index pairs (i, j) with i < j and 0 < ts[j] - ts[i] <= window.

    Pairs at the same timestamp are included (time_diff 0); see the note in
    `build_temporal_edges`.

    `ts` must be sorted ascending. Only sources in [lo_row, hi_row) are
    expanded, which is what keeps peak memory bounded: a dense market can
    generate tens of millions of pairs, and materialising every index at once
    is what makes the naive approach run out of RAM rather than merely be slow.

    Returns the source indices, target indices, and how many sources were
    truncated by `max_fanout`.
    """
    n = len(ts)
    hi_row = n if hi_row is None else min(hi_row, n)
    if n < 2 or lo_row >= hi_row:
        return np.empty(0, np.int64), np.empty(0, np.int64), 0

    rows = np.arange(lo_row, hi_row, dtype=np.int64)
    hi = np.searchsorted(ts, ts[rows] + window, side="right")
    counts = np.maximum(hi - rows - 1, 0)

    n_truncated = 0
    if max_fanout > 0:
        n_truncated = int((counts > max_fanout).sum())
        counts = np.minimum(counts, max_fanout)

    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), n_truncated

    src = np.repeat(rows, counts)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    within = np.arange(total, dtype=np.int64) - np.repeat(offsets, counts)
    tgt = src + 1 + within
    return src, tgt, n_truncated


def build_temporal_edges(
    trades: pd.DataFrame,
    window_seconds: int = 300,
    max_pairs_per_trade: int = 0,
    drop_crowd_timestamps: int = 0,
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    """Build directed temporal interaction edges for one market.

    An edge runs from the earlier trade to the later one when two *different*
    wallets trade within `window_seconds` of each other.

    Trades sharing a timestamp are linked with `time_diff == 0`, ordered by
    their position after a stable sort. This matches the original notebook
    implementation, and it matters: simultaneous fills are exactly the pattern
    the tightness screen is looking for, so dropping them would remove the
    signal. The consequence is that edge direction between simultaneous trades
    is an artifact of sort order, not of causality -- do not read it as one.

    Args:
        trades: canonical trade frame for a single market.
        window_seconds: the interaction window. Every downstream statistic
            depends on this value, so it belongs in config, not in code.
        max_pairs_per_trade: cap on how many partners one trade may generate.
            0 disables the cap.
        drop_crowd_timestamps: drop timestamps carrying more than this many
            trades before building. 0 disables. These are market-wide reaction
            moments where co-timing carries no relational information.
        chunk_size: pairs processed per batch, to bound peak memory.

    Returns:
        A frame matching `EDGE_COLUMNS`.
    """
    if trades.empty:
        return pd.DataFrame(columns=list(EDGE_COLUMNS)).pipe(validate_edges)

    market_ids = trades["market_id"].unique()
    if len(market_ids) > 1:
        raise ValueError(
            f"build_temporal_edges expects one market, got {len(market_ids)}. "
            "Use build_edges_multi_market() instead.")

    t = trades.sort_values(["timestamp", "wallet"], kind="mergesort").reset_index(drop=True)

    if drop_crowd_timestamps > 0:
        counts = t["timestamp"].value_counts()
        crowd = counts[counts > drop_crowd_timestamps].index
        if len(crowd):
            before = len(t)
            t = t[~t["timestamp"].isin(crowd)].reset_index(drop=True)
            log.info("dropped %d trades at %d crowd timestamps (%.1f%%)",
                     before - len(t), len(crowd), 100 * (before - len(t)) / before)

    if t.empty:
        # The crowd filter can legitimately remove everything when a market's
        # whole history sits inside a few burst moments. That is an answer, not
        # a crash: it means there is no non-crowd co-timing to study here.
        log.warning("no trades left after drop_crowd_timestamps=%d",
                    drop_crowd_timestamps)
        return pd.DataFrame(columns=list(EDGE_COLUMNS)).pipe(validate_edges)

    ts = t["timestamp"].to_numpy(np.int64)
    wallet = t["wallet"].to_numpy()
    side = t["side"].to_numpy()
    outcome = t["outcome"].to_numpy()
    size = t["size"].to_numpy(np.float64)
    price = t["price"].to_numpy(np.float64)

    frames = []
    truncated_total = 0
    n_rows = len(ts)
    # Row block size is derived from the pair budget so that a dense market and
    # a sparse one both stay inside roughly the same memory envelope.
    est_fanout = max(1, int(np.mean(np.searchsorted(ts, ts + window_seconds, side="right")
                                    - np.arange(n_rows))))
    row_block = max(1_000, min(n_rows, chunk_size // est_fanout))

    for lo in range(0, n_rows, row_block):
        s, g, n_trunc = _window_pairs(ts, window_seconds, max_pairs_per_trade,
                                      lo, lo + row_block)
        truncated_total += n_trunc
        if len(s) == 0:
            continue

        keep = wallet[s] != wallet[g]          # never link a wallet to itself
        s, g = s[keep], g[keep]
        if len(s) == 0:
            continue

        size_s, size_g = size[s], size[g]
        denom = np.maximum(size_s, size_g)
        ratio = np.where(denom > 0,
                         np.minimum(size_s, size_g) / np.where(denom > 0, denom, 1),
                         0.0)

        same_side = (side[s] == side[g]).astype(np.int8)
        same_outcome = (outcome[s] == outcome[g]).astype(np.int8)

        frames.append(pd.DataFrame({
            "market_id": market_ids[0],
            "source": wallet[s],
            "target": wallet[g],
            "timestamp": ts[g],
            "time_diff": (ts[g] - ts[s]).astype(np.float64),
            "same_side": same_side,
            "same_outcome": same_outcome,
            "same_behavior": (same_side & same_outcome).astype(np.int8),
            "price_diff": np.abs(price[s] - price[g]),
            "size_ratio": ratio,
            "source_size": size_s,
            "target_size": size_g,
            "source_price": price[s],
            "target_price": price[g],
        }))

    if truncated_total:
        log.warning(
            "%d trades exceeded max_pairs_per_trade=%d and were truncated; "
            "these are burst moments -- consider drop_crowd_timestamps instead",
            truncated_total, max_pairs_per_trade)

    if not frames:
        return pd.DataFrame(columns=list(EDGE_COLUMNS)).pipe(validate_edges)

    edges = pd.concat(frames, ignore_index=True)
    edges = edges.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    log.info("built %d temporal edges from %d trades (window=%ds)",
             len(edges), len(t), window_seconds)
    return validate_edges(edges)


def build_edges_multi_market(trades: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Build edges for several markets, keeping them strictly separate.

    Wallets are NOT linked across markets. Polymap's research framing is
    intra-market: within one market every wallet faces the same question and
    the same resolution, so co-timing is interpretable. Across markets two
    wallets trading at the same moment usually share nothing but a news cycle.

    Cross-market analysis is a different project with a different null model.
    If you want it, aggregate the per-market edge frames afterwards rather than
    relaxing this function.
    """
    out = []
    for market_id, group in trades.groupby("market_id", sort=False):
        log.info("market %s: %d trades", market_id, len(group))
        e = build_temporal_edges(group, **kwargs)
        if not e.empty:
            out.append(e)
    if not out:
        return pd.DataFrame(columns=list(EDGE_COLUMNS)).pipe(validate_edges)
    return pd.concat(out, ignore_index=True)
