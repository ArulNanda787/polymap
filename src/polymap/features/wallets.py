"""Wallet-level node features.

Deliberately contains no outcome information. Nothing here knows which side of
a market won, which is what makes the embedding-validation comparison in
`analysis/` meaningful rather than circular: if a feature leaked the result,
"embeddings predict behaviour" would be trivially true and worthless.

Features are computed per market by default. A wallet trading in two markets
gets two rows, because its behaviour in one says little about the other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import NODE_FEATURE_COLUMNS


def build_wallet_features(trades: pd.DataFrame, per_market: bool = True) -> pd.DataFrame:
    keys = ["market_id", "wallet"] if per_market else ["wallet"]
    t = trades.copy()
    t["is_buy"] = (t["side"] == "BUY").astype(int)

    g = t.groupby(keys, as_index=False).agg(
        trade_count=("wallet", "size"),
        buy_count=("is_buy", "sum"),
        avg_trade_size=("size", "mean"),
        median_trade_size=("size", "median"),
        total_size=("size", "sum"),
        avg_price=("price", "mean"),
        price_std=("price", "std"),
        unique_outcomes=("outcome", "nunique"),
        first_trade=("timestamp", "min"),
        last_trade=("timestamp", "max"),
    )

    g["sell_count"] = g["trade_count"] - g["buy_count"]
    g["buy_ratio"] = g["buy_count"] / g["trade_count"]
    g["sell_ratio"] = g["sell_count"] / g["trade_count"]
    g["price_std"] = g["price_std"].fillna(0.0)
    g["active_duration_hours"] = (g["last_trade"] - g["first_trade"]) / 3600.0
    g["trade_frequency"] = g["trade_count"] / g["active_duration_hours"].clip(lower=1 / 60)

    g["log_trade_count"] = np.log1p(g["trade_count"])
    g["log_avg_trade_size"] = np.log1p(g["avg_trade_size"])
    g["log_median_trade_size"] = np.log1p(g["median_trade_size"])
    g["log_active_duration"] = np.log1p(g["active_duration_hours"])

    missing = [c for c in NODE_FEATURE_COLUMNS if c not in g.columns]
    if missing:
        raise RuntimeError(f"feature builder did not produce {missing}")
    return g
