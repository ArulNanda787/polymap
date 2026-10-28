#!/usr/bin/env python3
"""Shrink the raw on-chain transfer table into a small file that is safe to share.

Reads the collector's normalised output (polygon_transfers.parquet / .csv) and writes:

  funding_edges.csv.gz     one row per (from_wallet, to_wallet, transfer_type, token_symbol):
                           n_transfers, first_ts, last_ts, total_value (token units), n_error_rows_dropped
  funder_fanout.csv        one row per from_wallet: how many distinct wallets it sent to
                           (among the queried wallets) -- shows the hub funders
  wallet_coverage.csv      one row per queried wallet: rows per endpoint type, first/last ts,
                           and whether it hit the 10k-row cap (likely truncated)

No private data beyond public addresses; typically a few MB even for millions of raw rows.

Usage (from backend/):
  python export_funding_summary.py data/raw/onchain_candidates/polygon_transfers.parquet \
      --outdir data/raw/onchain_candidates/summary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transfers")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    src = Path(a.transfers)
    out = Path(a.outdir) if a.outdir else src.parent / "summary"
    out.mkdir(parents=True, exist_ok=True)

    t = load(src)
    need = {"from_wallet", "to_wallet", "timestamp"}
    if not need <= set(t.columns):
        raise SystemExit(f"missing columns: {sorted(need - set(t.columns))}")
    n_raw = len(t)
    for c in ("from_wallet", "to_wallet", "queried_wallet"):
        if c in t:
            t[c] = t[c].astype(str).str.lower().str.strip()
    if "is_error" in t:
        err = t["is_error"].astype(str).isin(["1", "1.0", "True", "true"])
    else:
        err = pd.Series(False, index=t.index)
    n_err = int(err.sum())
    t = t[~err]

    # The same transfer is returned once per queried wallet; keep one copy.
    keys = [c for c in ("tx_hash", "from_wallet", "to_wallet", "transfer_type", "token_contract", "value_raw")
            if c in t.columns]
    if "tx_hash" in keys:
        t = t.drop_duplicates(subset=keys)
    else:
        t = t.drop_duplicates(subset=["from_wallet", "to_wallet", "timestamp"])

    t["transfer_type"] = t["transfer_type"] if "transfer_type" in t else "unknown"
    t["token_symbol"] = t["token_symbol"].fillna("").astype(str) if "token_symbol" in t else ""
    dec = pd.to_numeric(t["token_decimals"], errors="coerce").fillna(18) if "token_decimals" in t else 18
    raw = pd.to_numeric(t["value_raw"], errors="coerce").fillna(0.0) if "value_raw" in t else 0.0
    t["value"] = raw / np.power(10.0, dec)
    t["timestamp"] = pd.to_numeric(t["timestamp"], errors="coerce")

    edges = (t.groupby(["from_wallet", "to_wallet", "transfer_type", "token_symbol"], as_index=False)
             .agg(n_transfers=("timestamp", "size"), first_ts=("timestamp", "min"),
                  last_ts=("timestamp", "max"), total_value=("value", "sum")))
    edges.to_csv(out / "funding_edges.csv.gz", index=False)

    queried = set(t["queried_wallet"]) if "queried_wallet" in t else set(t["from_wallet"]) | set(t["to_wallet"])
    into = edges[edges["to_wallet"].isin(queried) & (edges["from_wallet"] != edges["to_wallet"])]
    fan = (into.groupby("from_wallet")
           .agg(distinct_recipients=("to_wallet", "nunique"), n_transfers=("n_transfers", "sum"),
                total_value=("total_value", "sum"),
                types=("transfer_type", lambda s: ",".join(sorted(set(s)))),
                tokens=("token_symbol", lambda s: ",".join(sorted(set(s))[:5])))
           .sort_values("distinct_recipients", ascending=False).reset_index())
    fan.to_csv(out / "funder_fanout.csv", index=False)

    if "queried_wallet" in t.columns:
        cov = (t.groupby(["queried_wallet", "transfer_type"]).size().unstack(fill_value=0))
        cov["total_rows"] = cov.sum(axis=1)
        span = t.groupby("queried_wallet")["timestamp"].agg(first_ts="min", last_ts="max")
        cov = cov.join(span)
        cov["maybe_capped_10k"] = (cov.drop(columns=["total_rows", "first_ts", "last_ts"]).max(axis=1) >= 10000)
        cov.reset_index().to_csv(out / "wallet_coverage.csv", index=False)

    print(f"raw rows {n_raw:,}; error rows dropped {n_err:,}; unique transfers {len(t):,}")
    print(f"funding_edges.csv.gz: {len(edges):,} rows; queried wallets: {len(queried):,}")
    print("\ntop funders by number of distinct queried wallets funded:")
    print(fan.head(15).to_string(index=False))
    for f in ("funding_edges.csv.gz", "funder_fanout.csv", "wallet_coverage.csv"):
        p = out / f
        if p.exists():
            print(f"{p}  {p.stat().st_size/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
