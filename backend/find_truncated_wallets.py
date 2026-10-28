#!/usr/bin/env python3
"""List wallets whose on-chain history was cut off by the Etherscan 10,000-row cap.

A wallet is flagged if EITHER
  (a) the collector's manifest lists it under truncated_address_endpoint_pages, OR
  (b) it has >= 9,000 ERC-20 rows and its history ends more than 7 days before the
      end of the data (the collector de-duplicates, so a capped wallet can land just
      under 10,000 rows and escape the manifest flag; this catches those).

Usage (from backend/):
  python find_truncated_wallets.py data/raw/onchain_rep_iran/polygon_transfers.parquet \
      --manifest data/raw/onchain_rep_iran/manifest.json \
      --candidates data/processed/rep_iran/candidate_pairs.parquet \
      --out truncated_rep_iran.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transfers")
    ap.add_argument("--manifest")
    ap.add_argument("--candidates", help="candidate_pairs.parquet, to report how many pairs are affected")
    ap.add_argument("--out", default="truncated_wallets.csv")
    ap.add_argument("--erc20-rows", type=int, default=9000)
    ap.add_argument("--gap-days", type=float, default=7.0)
    a = ap.parse_args()

    cols = ["queried_wallet", "transfer_type", "timestamp"]
    t = pd.read_parquet(a.transfers, columns=cols) if a.transfers.endswith(".parquet") \
        else pd.read_csv(a.transfers, usecols=cols)
    t["queried_wallet"] = t["queried_wallet"].astype(str).str.lower()
    t["timestamp"] = pd.to_numeric(t["timestamp"], errors="coerce")
    end = float(t["timestamp"].max())

    erc = t[t.transfer_type == "erc20"].groupby("queried_wallet").agg(
        erc20_rows=("timestamp", "size"), erc20_last_ts=("timestamp", "max"))
    erc["history_ends_days_before_end"] = ((end - erc["erc20_last_ts"]) / 86400).round(1)
    heur = erc[(erc.erc20_rows >= a.erc20_rows) & (erc.history_ends_days_before_end > a.gap_days)]

    flagged = {}
    if a.manifest and Path(a.manifest).exists():
        m = json.loads(Path(a.manifest).read_text())
        for r in m.get("truncated_address_endpoint_pages", []):
            flagged.setdefault(str(r["address"]).lower(), set()).add(r.get("action", "?"))

    rows = []
    for w in sorted(set(heur.index) | set(flagged)):
        rows.append({
            "queried_wallet": w,
            "in_manifest_cap_list": w in flagged,
            "manifest_endpoints": ",".join(sorted(flagged.get(w, []))),
            "erc20_rows": int(erc.erc20_rows.get(w, 0)),
            "history_ends_days_before_end": float(erc.history_ends_days_before_end.get(w, float("nan"))),
            "flagged_by_row_heuristic": w in heur.index,
        })
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    n_all = t["queried_wallet"].nunique()
    print(f"wallets with data: {n_all}; truncated: {len(out)} "
          f"({int(out.in_manifest_cap_list.sum()) if len(out) else 0} in manifest, "
          f"{int(out.flagged_by_row_heuristic.sum()) if len(out) else 0} by row heuristic)")
    if a.candidates and len(out):
        c = pd.read_parquet(a.candidates)
        bad = set(out.queried_wallet)
        hit = c.wallet_a.str.lower().isin(bad) | c.wallet_b.str.lower().isin(bad)
        print(f"candidate pairs involving a truncated wallet: {int(hit.sum())} of {len(c)} ({100*hit.mean():.0f}%)")
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
