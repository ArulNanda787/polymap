#!/usr/bin/env python3
"""Recover capped Polygon ERC-20 histories without rerunning other endpoints."""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from collect_polygon_transfers import (
    PAGE_SIZE,
    MAX_PAGES_PER_ENDPOINT,
    api_get,
)

OUTDIR = Path("data/raw/onchain/xi_2027_63d_p50")
RAW_DIR = OUTDIR / "raw_by_endpoint"
NORMALIZED = OUTDIR / "polygon_transfers.parquet"
MANIFEST = OUTDIR / "manifest.json"
REPORT_PATH = OUTDIR / "erc20_recovery_report.json"

TARGETS = [
    "0xd9fe6f0f41ca99fefa93c55d8890000e5a84db0d",
    "0x8b016395230775d54a3cd81d2e254653549124a9",
    "0xeed8c2be411025120fb511ea2332dd0a1f6135f0",
    "0x758dac51ba3cc9a246a79787da9df11c5f425d9e",
    "0x6b0ebbef2630b9a0da35594bf7ddd1911246e3ea",
]

MAX_DEPTH = 24


def fetch_complete_range(address, start, end, api_key, report, depth=0):
    """Fetch one block range; split it if pagination reaches the 10,000-row cap."""
    rows = []

    for page in range(1, MAX_PAGES_PER_ENDPOINT + 1):
        data = api_get(
            {
                "module": "account",
                "action": "tokentx",
                "address": address,
                "startblock": start,
                "endblock": end,
                "page": page,
                "offset": PAGE_SIZE,
                "sort": "asc",
            },
            api_key,
        )

        result = data.get("result", [])
        if not isinstance(result, list):
            raise RuntimeError(
                f"Unexpected API result for {address}, blocks {start}-{end}: "
                f"{result}"
            )

        rows.extend(result)

        if len(result) < PAGE_SIZE:
            return rows

    # All 10 pages were full: this range may be truncated.
    if start >= end or depth >= MAX_DEPTH:
        report["remaining_truncated_ranges"].append({
            "address": address,
            "start_block": start,
            "end_block": end,
            "depth": depth,
            "records_returned": len(rows),
            "reason": "Pagination cap reached; range could not be split further",
        })
        print(
            f"  WARNING: cap remains for blocks {start}-{end}",
            flush=True,
        )
        return rows

    midpoint = (start + end) // 2
    report["split_ranges"] += 1

    print(
        f"  Splitting capped range {start:,}-{end:,} "
        f"at {midpoint:,}",
        flush=True,
    )

    left = fetch_complete_range(
        address, start, midpoint, api_key, report, depth + 1
    )
    right = fetch_complete_range(
        address, midpoint + 1, end, api_key, report, depth + 1
    )

    return left + right


def normalize_record(tx, wallet):
    frm = str(tx.get("from", "")).lower()
    to = str(tx.get("to", "")).lower()

    if not frm or not to:
        return None

    try:
        timestamp = int(tx.get("timeStamp", 0))
        block = int(tx.get("blockNumber", 0) or 0)
    except (TypeError, ValueError):
        return None

    return {
        "from_wallet": frm,
        "to_wallet": to,
        "timestamp": timestamp,
        "tx_hash": tx.get("hash", ""),
        "block_number": block,
        "transfer_type": "erc20",
        "token_contract": str(tx.get("contractAddress", "")).lower(),
        "token_symbol": tx.get("tokenSymbol", ""),
        "token_decimals": tx.get("tokenDecimal", ""),
        "value_raw": tx.get("value", "0"),
        "queried_wallet": wallet,
        "is_error": tx.get("isError", "0"),
    }


def _configure(argv=None):
    """Optional CLI: --wallets-file <csv with queried_wallet column, or txt> --outdir <dir>."""
    import argparse
    global OUTDIR, RAW_DIR, NORMALIZED, MANIFEST, REPORT_PATH, TARGETS
    ap = argparse.ArgumentParser(description="Recover capped ERC-20 histories.")
    ap.add_argument("--wallets-file", help="csv (queried_wallet column) or one-address-per-line txt")
    ap.add_argument("--outdir", default=str(OUTDIR),
                    help="folder holding polygon_transfers.parquet and manifest.json "
                         f"(default: {OUTDIR})")
    a = ap.parse_args(argv)
    OUTDIR = Path(a.outdir)
    RAW_DIR = OUTDIR / "raw_by_endpoint"
    NORMALIZED = OUTDIR / "polygon_transfers.parquet"
    MANIFEST = OUTDIR / "manifest.json"
    REPORT_PATH = OUTDIR / "erc20_recovery_report.json"
    if a.wallets_file:
        wf = Path(a.wallets_file)
        if wf.suffix == ".csv":
            df = pd.read_csv(wf)
            col = "queried_wallet" if "queried_wallet" in df.columns else df.columns[0]
            TARGETS = [w.strip().lower() for w in df[col].astype(str)]
        else:
            TARGETS = [w.strip().lower() for w in wf.read_text().split() if w.strip()]
    print(f"Recovering {len(TARGETS)} wallets in {OUTDIR}", flush=True)


def main():
    _configure()
    api_key = os.getenv("ETHERSCAN_API_KEY", "").strip()
    if not api_key:
        sys.exit("ERROR: ETHERSCAN_API_KEY is not set in this terminal.")

    if not NORMALIZED.exists() or not MANIFEST.exists():
        sys.exit("ERROR: Existing Parquet or manifest is missing.")

    manifest = json.loads(MANIFEST.read_text())
    start_block = int(manifest["start_block"])
    end_block = int(manifest["end_block"])

    existing = pd.read_parquet(NORMALIZED)
    required = {"queried_wallet", "transfer_type"}
    if not required.issubset(existing.columns):
        sys.exit(f"ERROR: Parquet is missing columns: {required - set(existing.columns)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    backup_path = NORMALIZED.with_suffix(".parquet.pre_recovery")
    if not backup_path.exists():
        existing.to_parquet(backup_path, index=False)
        print(f"Created dataset backup: {backup_path}", flush=True)

    report = {
        "start_block": start_block,
        "end_block": end_block,
        "target_wallets": TARGETS,
        "wallet_results": {},
        "split_ranges": 0,
        "remaining_truncated_ranges": [],
    }

    # Preserve prior recovery details if this script is rerun.
    if REPORT_PATH.exists():
        try:
            old_report = json.loads(REPORT_PATH.read_text())
            report["previous_report"] = old_report
        except (OSError, json.JSONDecodeError):
            pass

    new_rows = []

    for index, wallet in enumerate(TARGETS, 1):
        print(f"\n[{index}/{len(TARGETS)}] Recovering {wallet}", flush=True)

        records = fetch_complete_range(
            wallet, start_block, end_block, api_key, report
        )

        raw_path = RAW_DIR / f"{wallet}_tokentx.json"
        if raw_path.exists():
            backup_raw = raw_path.with_suffix(".json.pre_recovery")
            if not backup_raw.exists():
                raw_path.replace(backup_raw)

        raw_path.write_text(json.dumps(records, indent=2))

        wallet_rows = [
            row for tx in records
            if (row := normalize_record(tx, wallet)) is not None
        ]
        new_rows.extend(wallet_rows)

        report["wallet_results"][wallet] = {
            "api_records": len(records),
            "normalized_records": len(wallet_rows),
        }

        print(
            f"  API records: {len(records):,}; "
            f"normalized: {len(wallet_rows):,}",
            flush=True,
        )

        # Checkpoint after each wallet so interruption does not lose completed work.
        combined = pd.concat(
            [existing, pd.DataFrame(new_rows)],
            ignore_index=True,
        ).drop_duplicates()

        combined.to_parquet(NORMALIZED, index=False)

        report["normalized_rows_after_checkpoint"] = len(combined)
        REPORT_PATH.write_text(json.dumps(report, indent=2))

    # Final deduplication and report.
    combined = pd.concat(
        [existing, pd.DataFrame(new_rows)],
        ignore_index=True,
    ).drop_duplicates()

    combined.to_parquet(NORMALIZED, index=False)

    report["existing_rows_before_recovery"] = len(existing)
    report["recovered_normalized_rows_before_dedup"] = len(new_rows)
    report["final_normalized_rows"] = len(combined)
    report["final_duplicate_rows_removed"] = (
        len(existing) + len(new_rows) - len(combined)
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\nRecovery finished.", flush=True)
    print(f"Dataset: {NORMALIZED}")
    print(f"Backup:  {backup_path}")
    print(f"Report:  {REPORT_PATH}")
    print(f"Final rows: {len(combined):,}")

    if report["remaining_truncated_ranges"]:
        print(
            "WARNING: Some ranges remain capped. Check "
            f"{REPORT_PATH} before treating the collection as complete."
        )
    else:
        print("No ranges remained at the pagination cap.")


if __name__ == "__main__":
    main()
