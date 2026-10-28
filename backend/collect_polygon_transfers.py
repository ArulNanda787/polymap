#!/usr/bin/env python3
"""Collect Polygon PoS transaction and ERC-20 transfer records for Polymarket wallets.

Reads the wallet/time window from a Polymarket trades Parquet file and writes raw
API records plus a normalized transfer file suitable for Polymap funding validation.
Uses Etherscan API V2, chain ID 137. Free-tier list endpoints are paginated at 1,000 rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

API_URL = "https://api.etherscan.io/v2/api"
CHAIN_ID = 137
PAGE_SIZE = 1000
MAX_PAGES_PER_ENDPOINT = 10  # Free-tier history cap; flag addresses hitting this cap.
MIN_CALL_INTERVAL = 0.26  # Stay below the stated 5 calls/second limit.
_last_call = 0.0


def api_get(params: dict[str, Any], api_key: str, retries: int = 8) -> dict[str, Any]:
    global _last_call
    full = {"chainid": CHAIN_ID, "apikey": api_key, **params}
    for attempt in range(retries):
        wait = MIN_CALL_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            response = requests.get(API_URL, params=full, timeout=45)
            _last_call = time.monotonic()
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(2 ** attempt, 30))
                continue
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"API request failed after retries: {exc}") from exc
            time.sleep(min(2 ** attempt, 30))
            continue

        result = data.get("result")
        message = str(data.get("message", ""))
        # Etherscan's proxy module returns JSON-RPC payloads without status="1".
        if data.get("jsonrpc") and result is not None:
            return data
        if data.get("status") == "1":
            return data
        # List endpoints commonly use status=0 for an empty result set. This is
        # a valid outcome for a wallet with no transactions of that type, not an API failure.
        if "no transactions found" in (message + " " + str(result)).lower() and (
            result == [] or isinstance(result, str)
        ):
            return {"status": "1", "message": "No transactions found", "result": []}
        err = f"Etherscan API response: {data}"
        if any(s in (message + " " + str(result)).lower() for s in
               ("rate limit", "max rate", "temporarily unavailable", "timeout")):
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 30))
                continue
        raise RuntimeError(err)
    raise RuntimeError("Etherscan API request exhausted retries")


def block_for_timestamp(timestamp: int, api_key: str, closest: str) -> int:
    data = api_get({"module": "block", "action": "getblocknobytime",
                    "timestamp": timestamp, "closest": closest}, api_key)
    try:
        return int(data["result"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not resolve {closest} block for timestamp {timestamp}: {data}") from exc


def latest_block_number(api_key: str) -> int:
    """Return the current Polygon head; padded trade windows can extend into the future."""
    data = api_get({"module": "proxy", "action": "eth_blockNumber"}, api_key)
    try:
        return int(data["result"], 16)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not resolve latest Polygon block: {data}") from exc


def fetch_address_endpoint(address: str, action: str, start_block: int, end_block: int,
                           api_key: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in range(1, MAX_PAGES_PER_ENDPOINT + 1):
        data = api_get({
            "module": "account", "action": action, "address": address,
            "startblock": start_block, "endblock": end_block,
            "page": page, "offset": PAGE_SIZE, "sort": "asc",
        }, api_key)
        result = data.get("result", [])
        if not isinstance(result, list):
            raise RuntimeError(f"Unexpected result for {action} {address}: {result}")
        rows.extend(result)
        if len(result) < PAGE_SIZE:
            break
        if page == MAX_PAGES_PER_ENDPOINT:
            report["truncated_address_endpoint_pages"].append({
                "address": address, "action": action,
                "reason": f"Reached {MAX_PAGES_PER_ENDPOINT} pages x {PAGE_SIZE} rows",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True, help="Fresh trades Parquet file")
    parser.add_argument("--outdir", default="data/raw/onchain", help="Output directory")
    parser.add_argument("--wallets-file",
                        help="Text file with one wallet per line (e.g. data/processed/<run>/"
                             "candidate_wallets.txt). Only these wallets are fetched; "
                             "--trades is still used for the time window.")
    parser.add_argument("--limit-wallets", type=int, default=0,
                        help="Optional small test limit (0 means all wallets)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip wallets already recorded as completed in progress.json")
    args = parser.parse_args()

    api_key = os.getenv("ETHERSCAN_API_KEY", "").strip()
    if not api_key:
        print("ERROR: ETHERSCAN_API_KEY is not set in this terminal.", file=sys.stderr)
        return 2

    trades_path = Path(args.trades)
    if not trades_path.exists():
        print(f"ERROR: trades file not found: {trades_path}", file=sys.stderr)
        return 2
    trades = pd.read_parquet(trades_path)
    needed = {"wallet", "timestamp"}
    if not needed.issubset(trades.columns) or trades.empty:
        print(f"ERROR: trades file must be non-empty and contain {sorted(needed)}", file=sys.stderr)
        return 2

    trades["wallet"] = trades["wallet"].astype(str).str.lower().str.strip()
    trades = trades[trades["wallet"].str.fullmatch(r"0x[a-f0-9]{40}", na=False)]
    wallets = sorted(trades["wallet"].unique().tolist())
    if args.wallets_file:
        wf = Path(args.wallets_file)
        if not wf.exists():
            print(f"ERROR: wallets file not found: {wf}", file=sys.stderr)
            return 2
        wanted = {w.strip().lower() for w in wf.read_text().split() if w.strip()}
        unknown = wanted - set(wallets)
        if unknown:
            print(f"NOTE: {len(unknown)} wallets in {wf.name} are not in the trades file; "
                  f"fetching them anyway", file=sys.stderr)
        wallets = sorted(wanted)
    if args.limit_wallets:
        wallets = wallets[:args.limit_wallets]
    min_ts, max_ts = int(pd.to_numeric(trades["timestamp"]).min()), int(pd.to_numeric(trades["timestamp"]).max())
    # Expand the observed trade window by 24h on each side to catch nearby funding.
    start_ts, end_ts = min_ts - 86400, max_ts + 86400
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    progress_path = outdir / "progress.json"
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
    else:
        progress = {"completed_wallets": [], "truncated_address_endpoint_pages": [], "errors": []}
    completed = set(progress.get("completed_wallets", []))

    print(f"Trades file: {trades_path}")
    print(f"Wallets to collect: {len(wallets):,}; already completed: {len(completed):,}")
    print("Trade timestamp window (UTC):",
          datetime.fromtimestamp(min_ts, timezone.utc).isoformat(), "to",
          datetime.fromtimestamp(max_ts, timezone.utc).isoformat())
    start_block = block_for_timestamp(start_ts, api_key, "before")
    # The trade window is padded by 24 hours, so its end can be in the future.
    # Etherscan's timestamp-to-block endpoint rejects future timestamps; use the
    # current chain head as the upper bound instead.
    end_block = latest_block_number(api_key)
    if start_block > end_block:
        raise RuntimeError(
            f"Start block {start_block} is after current chain head {end_block}; "
            "check the trade timestamps and Polygon API response."
        )
    print(f"Polygon block range (start includes 24h lookback; end is current chain head): "
          f"{start_block:,}..{end_block:,}")

    normalized_path = outdir / "polygon_transfers.parquet"
    raw_dir = outdir / "raw_by_endpoint"
    raw_dir.mkdir(exist_ok=True)
    if normalized_path.exists() and args.resume:
        existing = pd.read_parquet(normalized_path)
        normalized_rows = existing.to_dict("records")
    else:
        normalized_rows: list[dict[str, Any]] = []

    endpoints = [("txlist", "native_tx"), ("txlistinternal", "internal_native"),
                 ("tokentx", "erc20")]
    try:
        for i, wallet in enumerate(wallets, 1):
            if wallet in completed:
                continue
            wallet_rows: list[dict[str, Any]] = []
            print(f"[{i}/{len(wallets)}] {wallet}", flush=True)
            for action, transfer_type in endpoints:
                records = fetch_address_endpoint(wallet, action, start_block, end_block,
                                                 api_key, progress)
                pd.DataFrame(records).to_json(
                    raw_dir / f"{wallet}_{action}.json", orient="records", indent=2)
                for tx in records:
                    frm = str(tx.get("from", "")).lower()
                    to = str(tx.get("to", "")).lower()
                    if not frm or not to:
                        continue
                    try:
                        ts = int(tx.get("timeStamp", 0))
                    except (TypeError, ValueError):
                        continue
                    # Retain only rows in the padded time window even if API filtering is broad.
                    if not (start_ts <= ts <= end_ts):
                        continue
                    wallet_rows.append({
                        "from_wallet": frm,
                        "to_wallet": to,
                        "timestamp": ts,
                        "tx_hash": tx.get("hash", ""),
                        "block_number": int(tx.get("blockNumber", 0) or 0),
                        "transfer_type": transfer_type,
                        "token_contract": str(tx.get("contractAddress", "")).lower(),
                        "token_symbol": tx.get("tokenSymbol", "POL" if transfer_type != "erc20" else ""),
                        "token_decimals": tx.get("tokenDecimal", "18" if transfer_type != "erc20" else ""),
                        "value_raw": tx.get("value", "0"),
                        "queried_wallet": wallet,
                        "is_error": tx.get("isError", "0"),
                    })
            normalized_rows.extend(wallet_rows)
            completed.add(wallet)
            progress["completed_wallets"] = sorted(completed)
            progress["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
            # Checkpoint each wallet so an interrupted run can be resumed.
            pd.DataFrame(normalized_rows).drop_duplicates().to_parquet(normalized_path, index=False)
            progress_path.write_text(json.dumps(progress, indent=2))
            if i % 25 == 0:
                print(f"Checkpoint: {len(completed):,}/{len(wallets):,} wallets; "
                      f"{len(normalized_rows):,} transfer rows", flush=True)
    except KeyboardInterrupt:
        progress["completed_wallets"] = sorted(completed)
        progress["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        progress_path.write_text(json.dumps(progress, indent=2))
        print("Interrupted; progress saved. Re-run with --resume to continue.", file=sys.stderr)
        return 130
    except Exception as exc:
        progress["completed_wallets"] = sorted(completed)
        progress["errors"].append({"error": str(exc), "at_utc": datetime.now(timezone.utc).isoformat()})
        progress_path.write_text(json.dumps(progress, indent=2))
        print(f"ERROR: {exc}\nProgress saved to {progress_path}", file=sys.stderr)
        return 1

    final = pd.DataFrame(normalized_rows).drop_duplicates()
    final.to_parquet(normalized_path, index=False)
    final.to_csv(outdir / "polygon_transfers.csv", index=False)
    manifest = {
        "chain": "Polygon PoS", "chain_id": CHAIN_ID,
        "trades_file": str(trades_path), "n_trade_rows": int(len(trades)),
        "n_wallets_requested": len(wallets), "n_wallets_completed": len(completed),
        "trade_window_utc": {
            "start": datetime.fromtimestamp(min_ts, timezone.utc).isoformat(),
            "end": datetime.fromtimestamp(max_ts, timezone.utc).isoformat(),
            "queried_start_with_padding": datetime.fromtimestamp(start_ts, timezone.utc).isoformat(),
            "queried_end_with_padding": datetime.fromtimestamp(end_ts, timezone.utc).isoformat(),
        },
        "start_block": start_block, "end_block": end_block,
        "n_normalized_transfer_rows": int(len(final)),
        "truncated_address_endpoint_pages": progress.get("truncated_address_endpoint_pages", []),
        "errors": progress.get("errors", []),
        "limitations": [
            "Only Polygon PoS was queried.",
            "Free-tier address endpoint results are paginated at 1,000 rows and capped to 10 pages per endpoint here.",
            "The collection includes native transactions, internal native transfers by address, and ERC-20 transfers; it does not include every NFT transfer or cross-chain funding path.",
            "Shared exchange/bridge/router funding is not proof of common ownership.",
        ],
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE: {len(final):,} normalized transfer rows")
    print(f"Parquet: {normalized_path}")
    print(f"CSV:     {outdir / 'polygon_transfers.csv'}")
    print(f"Manifest: {outdir / 'manifest.json'}")
    if manifest["truncated_address_endpoint_pages"]:
        print(f"WARNING: {len(manifest['truncated_address_endpoint_pages'])} address/endpoint queries hit the pagination cap; inspect manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
