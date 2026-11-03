"""Market discovery and trade retrieval, with immutable caching.

**The caching is not a speed optimisation, it is the reproducibility fix.**
The Data API caps a market at ~10,000 trades and returns a *different* slice on
repeated calls: an earlier pull of market 665374 covered 19 Jul to 20 Sep 2026
with 2,358 wallets, while a later identical request returned 10 to 26 Sep with
roughly 1,351. Any result computed from a live fetch is therefore unrepeatable.

So the first fetch writes raw trades to `data/raw/<market>.parquet` alongside a
manifest carrying a SHA-256 of the file, the row count, the observed time span
and the fetch time. Every later run reads the cached file unless you explicitly
pass `refresh=True`. The hash is what lets you state in a paper that a result
came from a specific artifact, and lets a reader check they have the same one.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from .client import ApiError, Client
import pandas as pd
import time
from ..config import Config
from .schema import TRADE_COLUMNS, TRADE_FIELD_MAP, validate_trades
from .client import ApiError, Client
log = logging.getLogger(__name__)


# ------------------------------------------------------------------ markets --
def _maybe_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def parse_market(raw: dict) -> dict | None:
    """Normalise one Gamma market record.

    Gamma returns `outcomes` and `clobTokenIds` as JSON-encoded arrays whose
    entries line up by index.
    """
    outcomes = _maybe_json(raw.get("outcomes"))
    token_ids = _maybe_json(raw.get("clobTokenIds"))
    prices = _maybe_json(raw.get("outcomePrices"))
    if not outcomes or not token_ids or len(outcomes) != len(token_ids):
        return None

    winners = {}
    resolved = False
    if prices and len(prices) == len(token_ids):
        try:
            values = [float(p) for p in prices]
        except (TypeError, ValueError):
            return None
        # A settled market pins prices at ~1 and ~0. `closed: true` alone does
        # not mean resolved, and a mid-range price would poison any PnL work.
        resolved = all(p <= 0.05 or p >= 0.95 for p in values) and sum(
            1 for p in values if p > 0.5) == 1
        winners = {t: (1.0 if p > 0.5 else 0.0) for t, p in zip(token_ids, values)}

    try:
        volume = float(raw.get("volumeNum") or raw.get("volume") or 0)
    except (TypeError, ValueError):
        volume = 0.0

    return {
        "market_id": raw.get("conditionId"),
        "gamma_id": str(raw.get("id", "")),
        "slug": raw.get("slug"),
        "question": raw.get("question"),
        "token_ids": token_ids,
        "outcomes": outcomes,
        "winner_by_token": winners,
        "resolved": resolved,
        "closed": bool(raw.get("closed")),
        "volume": volume,
        "end_date": raw.get("endDate"),
    }


def discover_markets(cfg: Config, min_volume: float = 0.0, closed: bool | None = None,
                     limit: int = 100, client: Client | None = None) -> pd.DataFrame:
    """Scan Gamma and rank markets locally.

    Gamma's `order=volume` parameter is silently ignored: asking for the
    highest-volume closed market returns one with about $999 of volume. So we
    page a wide net and sort ourselves.
    """
    client = client or Client(cfg.api)
    params: dict = {}
    if closed is not None:
        params["closed"] = "true" if closed else "false"

    rows, seen = [], set()
    for raw in client.paginate(f"{cfg.api.gamma_base}/markets", params):
        m = parse_market(raw)
        if not m or not m["market_id"] or m["market_id"] in seen:
            continue
        seen.add(m["market_id"])
        if m["volume"] < min_volume:
            continue
        rows.append(m)

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("no markets matched (min_volume=%s, closed=%s)", min_volume, closed)
        return df
    return df.sort_values("volume", ascending=False).head(limit).reset_index(drop=True)


def find_market(cfg: Config, market_key: str, client: Client | None = None) -> dict | None:
    """Look up one market by condition id, Gamma numeric id, or slug."""
    client = client or Client(cfg.api)
    for params in ({"condition_ids": market_key}, {"id": market_key}, {"slug": market_key}):
        try:
            batch = client.get(f"{cfg.api.gamma_base}/markets", {**params, "limit": 5})
        except Exception:
            continue
        for raw in batch or []:
            m = parse_market(raw)
            if not m:
                continue
            # Gamma ignores filters it does not recognise and returns a default
            # page, so confirm the result actually matches what we asked for.
            if market_key in (m["market_id"], m["gamma_id"], m["slug"]):
                return m
    return None


# ------------------------------------------------------------------- trades --
def normalise_trades(raw: list[dict], market_id: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=list(TRADE_COLUMNS))
    df = pd.DataFrame(raw)

    out = pd.DataFrame(index=df.index)
    for name, candidates in TRADE_FIELD_MAP.items():
        for c in candidates:
            if c in df.columns:
                out[name] = df[c]
                break
    out["market_id"] = market_id
    for col in ("outcome", "tx_hash"):
        if col not in out:
            out[col] = ""

    required = ["wallet", "token_id", "side", "size", "price", "timestamp"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"trade payload is missing {missing}. Observed keys: "
            f"{sorted(df.columns.tolist())[:20]}. Add the new names to "
            f"TRADE_FIELD_MAP in polymap/data/schema.py.")

    out = out.dropna(subset=required)
    out["wallet"] = out["wallet"].astype(str).str.lower()
    out["side"] = out["side"].astype(str).str.upper().str[:1].map(
        {"B": "BUY", "S": "SELL"}).fillna("BUY")
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    return validate_trades(out[list(TRADE_COLUMNS)])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


PLACEHOLDER_IDS = {"0x...", "0x", "...", "", "REPLACE_ME", "<market_id>"}


def _check_market_id(market_id: str) -> None:
    """Fail loudly on a config placeholder.

    Left unchecked, a placeholder produces an empty result set and a mild
    warning, and the real cause (an unedited config two directories away)
    stays invisible while the error surfaces at a later stage.
    """
    mid = (market_id or "").strip()
    if mid in PLACEHOLDER_IDS or mid.startswith("0x...") or len(mid) < 10:
        raise ValueError(
            f"'{market_id}' is not a real market id -- your config still has the "
            f"placeholder.\n"
            f"  A Polymarket conditionId is a 66-character hex string.\n"
            f"  Find one with:  polymap markets --min-volume 200000\n"
            f"  Then edit the `markets:` list in your config file.")


# Data API: rows past offset 10,000 are rejected, but every start/end window
# has its own offset budget. So deep history = many small windows.
DATA_API_MAX_OFFSET = 10_000
_PROBE_OFFSET = DATA_API_MAX_OFFSET - 1     # a row here means the window holds >= 10,000


def _trade_params(market_id: str, start: int, end: int, limit: int, offset: int) -> dict:
    return {"market": market_id, "takerOnly": "false",
            "start": start, "end": end, "limit": limit, "offset": offset}


def _page_window(client: Client, cfg: Config, market_id: str, start: int, end: int) -> list[dict]:
    """Page one window to exhaustion. Only called when it holds < 10,000 rows."""
    url = f"{cfg.api.data_base}/trades"
    rows: list[dict] = []
    offset = 0
    while offset <= DATA_API_MAX_OFFSET:
        try:
            batch = client.get(url, _trade_params(market_id, start, end,
                                                  cfg.api.page_size, offset))
        except ApiError as e:
            if e.status in (400, 422) and "offset" in (e.body or "").lower():
                break
            raise
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)    # advance by what we got, in case the server clamps page size
    return rows


def _collect_windows(client: Client, cfg: Config, market_id: str, start: int, end: int,
                     min_width: int = 2) -> list[dict]:
    """All trades in [start, end], splitting time until each window fits the cap.

    One cheap probe (limit=1 at offset 9,999) decides whether a window is too
    big, so oversized windows are split without paging through them first.
    Halves overlap by one second so boundary trades are never lost whichever way
    the API treats inclusive bounds; the caller de-duplicates.
    """
    url = f"{cfg.api.data_base}/trades"
    too_big = bool(client.get(url, _trade_params(market_id, start, end, 1, _PROBE_OFFSET)))
    if too_big and end - start > min_width:
        mid = (start + end) // 2
        log.info("window %s..%s holds >=10k trades, splitting",
                 datetime.fromtimestamp(start, timezone.utc).strftime("%m-%d %H:%M"),
                 datetime.fromtimestamp(end, timezone.utc).strftime("%m-%d %H:%M"))
        return (_collect_windows(client, cfg, market_id, start, mid, min_width)
                + _collect_windows(client, cfg, market_id, mid, end, min_width))
    if too_big:
        log.warning("a %ds window still holds >=10k trades; keeping the first 10k only", end - start)
    rows = _page_window(client, cfg, market_id, start, end)
    log.info("window %s..%s: %d trades",
             datetime.fromtimestamp(start, timezone.utc).strftime("%m-%d %H:%M"),
             datetime.fromtimestamp(end, timezone.utc).strftime("%m-%d %H:%M"), len(rows))
    return rows


def fetch_trades(cfg: Config, market_id: str, refresh: bool = False,
                 client: Client | None = None) -> pd.DataFrame:
    """Return trades for one market, from cache when available.

    With `cfg.history_days > 0` the last N days are fetched by walking time
    windows (not limited by the 10,000-row offset cap) and cached under a
    separate name, so it never collides with an older newest-first slice.

    Set `refresh=True` only when you intend to replace the archived slice, and
    understand that doing so changes every downstream result.
    """
    _check_market_id(market_id)
    cfg.ensure_dirs()
    safe = market_id.replace("/", "_")[:64]
    windowed = cfg.history_days > 0
    tag = f"_d{cfg.history_days}" if windowed else ""
    cache = cfg.raw_dir / f"trades_{safe}{tag}.parquet"
    manifest_path = cfg.raw_dir / f"trades_{safe}{tag}.manifest.json"

    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        log.info("cache hit: %s (%d trades)", cache.name, len(df))
        return validate_trades(df)

    client = client or Client(cfg.api)
    win_start = win_end = None
    if windowed:
        win_end = int(time.time())
        win_start = win_end - cfg.history_days * 86_400
        log.info("fetching %d days of trades for %s by time window", cfg.history_days, market_id)
        raw = _collect_windows(client, cfg, market_id, win_start, win_end)
    else:
        log.info("fetching trades for %s (cap %d)", market_id, cfg.max_trades_per_market)
        raw = list(client.paginate(f"{cfg.api.data_base}/trades",
                                   {"market": market_id, "takerOnly": False},
                                   max_items=cfg.max_trades_per_market))
    df = normalise_trades(raw, market_id)
    if df.empty:
        log.warning("no trades returned for %s", market_id)
        return df

    if windowed:    # drop the 1-second overlaps between window halves
        before = len(df)
        df = df.drop_duplicates(subset=["wallet", "token_id", "side", "size", "price",
                                        "timestamp", "tx_hash"])
        log.info("de-duplicated window overlaps: %d -> %d rows", before, len(df))

    df = df.sort_values("timestamp").reset_index(drop=True)
    truncated = False
    if windowed and cfg.max_trades_per_market > 0 and len(df) > cfg.max_trades_per_market:
        df = df.tail(cfg.max_trades_per_market).reset_index(drop=True)   # keep the newest
        truncated = True
    df.to_parquet(cache, index=False)

    manifest = {
        "market_id": market_id,
        "mode": "windowed" if windowed else "newest_first",
        "history_days": cfg.history_days,
        "window_start_utc": None if win_start is None else datetime.fromtimestamp(win_start, timezone.utc).isoformat(),
        "window_end_utc": None if win_end is None else datetime.fromtimestamp(win_end, timezone.utc).isoformat(),
        "n_trades": int(len(df)),
        "n_wallets": int(df["wallet"].nunique()),
        "first_trade_utc": datetime.fromtimestamp(
            int(df["timestamp"].min()), timezone.utc).isoformat(),
        "last_trade_utc": datetime.fromtimestamp(
            int(df["timestamp"].max()), timezone.utc).isoformat(),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "hit_api_cap": bool(truncated if windowed else len(df) >= cfg.max_trades_per_market),
        "sha256": _sha256(cache),
        "file": cache.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if manifest["hit_api_cap"]:
        log.warning(
            "market %s hit the %d-trade cap: this slice is a truncated "
            "sample, not the market's full history. Say so in any writeup.",
            market_id, cfg.max_trades_per_market)
    log.info("cached %d trades (%s -> %s) -> %s (sha256 %s...)", len(df),
             manifest["first_trade_utc"][:16], manifest["last_trade_utc"][:16],
             cache.name, manifest["sha256"][:12])
    return df


def fetch_many(cfg: Config, market_ids: list[str] | None = None,
               refresh: bool = False) -> pd.DataFrame:
    """Fetch several markets and return one frame keyed by `market_id`."""
    ids = market_ids if market_ids is not None else cfg.markets
    if not ids:
        raise ValueError("no markets configured. Set `markets:` in your config "
                         "or pass market_ids.")
    client = Client(cfg.api)
    frames, failures = [], []
    for mid in ids:
        try:
            df = fetch_trades(cfg, mid, refresh=refresh, client=client)
            if not df.empty:
                frames.append(df)
        except Exception as e:  # one bad market should not kill the batch
            log.error("market %s failed: %s", mid, e)
            failures.append((mid, str(e).split(chr(10))[0]))
    if not frames:
        detail = "".join(f"\n    {m}: {why}" for m, why in failures)
        raise ValueError(
            f"no trades fetched for any of {len(ids)} market(s).{detail}\n"
            f"  Check the `markets:` list in your config holds real condition "
            f"ids, and that you can reach the API from this network.")
    return pd.concat(frames, ignore_index=True)
