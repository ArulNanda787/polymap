"""Canonical column names and dataframe validation.

Every module in Polymap reads and writes these names. The Polymarket API uses
different ones (`proxyWallet`, `asset`, `conditionId`), and those names have
changed between API versions before. Normalising once at the boundary means an
upstream rename touches exactly one file: `polymap/data/trades.py`.

Validation is cheap and catches the failure mode that costs the most time --
a stage that runs happily on a dataframe missing a column it silently expected.
"""
from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------- trades
TRADE_COLUMNS = {
    "market_id": "str",      # our market key (condition id)
    "wallet": "str",         # proxy wallet address, lowercased
    "token_id": "str",       # outcome token
    "outcome": "str",        # human-readable outcome label
    "side": "str",           # BUY or SELL
    "size": "float64",
    "price": "float64",
    "timestamp": "int64",    # unix seconds
    "tx_hash": "str",
}

# Maps our names -> candidate API keys, first match wins.
# Add alternatives here when the API renames something.
TRADE_FIELD_MAP = {
    "wallet": ["proxyWallet", "proxy_wallet", "maker", "user", "wallet"],
    "token_id": ["asset", "asset_id", "tokenId", "token_id"],
    "outcome": ["outcome", "outcomeLabel"],
    "side": ["side", "Side", "direction"],
    "size": ["size", "amount", "quantity", "shares"],
    "price": ["price", "Price", "avgPrice"],
    "timestamp": ["timestamp", "matchTime", "created_at", "time"],
    "tx_hash": ["transactionHash", "tx_hash", "hash"],
    "market_id": ["conditionId", "condition_id", "market"],
}

# ---------------------------------------------------------------------- edges
EDGE_COLUMNS = {
    "market_id": "str",
    "source": "str",
    "target": "str",
    "timestamp": "int64",     # timestamp of the LATER trade
    "time_diff": "float64",   # seconds between the two trades
    "same_side": "int8",
    "same_outcome": "int8",
    "same_behavior": "int8",
    "price_diff": "float64",
    "size_ratio": "float64",
    "source_size": "float64",
    "target_size": "float64",
    "source_price": "float64",
    "target_price": "float64",
}

# Features fed to the model. price_diff is deliberately EXCLUDED: in a binary
# market it is almost a restatement of same_outcome (mean 0.875 when outcomes
# differ, 0.004 when they match), so including it adds redundancy, not signal.
EDGE_FEATURE_COLUMNS = ["time_diff", "same_side", "same_outcome", "size_ratio"]

NODE_FEATURE_COLUMNS = [
    "log_trade_count",
    "buy_ratio",
    "sell_ratio",
    "log_avg_trade_size",
    "log_median_trade_size",
    "avg_price",
    "price_std",
    "unique_outcomes",
    "log_active_duration",
]


class SchemaError(ValueError):
    """Raised when a dataframe does not match the expected schema."""


def validate(df: pd.DataFrame, expected: dict[str, str], name: str) -> pd.DataFrame:
    """Check required columns exist, then coerce dtypes.

    Raises early with a message naming the stage, rather than letting a missing
    column surface three stages later as an incomprehensible KeyError.
    """
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{name}: missing columns {missing}. "
            f"Present: {sorted(df.columns.tolist())[:15]}"
        )
    out = df.copy()
    for col, dtype in expected.items():
        try:
            out[col] = out[col].astype(dtype)
        except (TypeError, ValueError) as e:
            raise SchemaError(f"{name}: column '{col}' will not cast to {dtype}: {e}")
    return out


def validate_trades(df: pd.DataFrame) -> pd.DataFrame:
    return validate(df, TRADE_COLUMNS, "trades")


def validate_edges(df: pd.DataFrame) -> pd.DataFrame:
    return validate(df, EDGE_COLUMNS, "edges")
