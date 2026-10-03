"""HTTP client for the Polymarket APIs.

Two behaviours here exist because of specific failures we hit:

1. **No custom User-Agent.** Cloudflare fronts these hosts and treats unusual
   agent strings as a bot signal. A plain `requests` default passes where a
   bespoke one gets challenged.

2. **4xx fails immediately and loudly.** Retrying a 4xx wastes time and buries
   the real message. HTTP 451 (blocked for legal reasons) and 422 (`offset too
   large`) both arrive as 4xx, and both need to be seen rather than swallowed
   into a generic timeout after four attempts.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

from ..config import ApiConfig

log = logging.getLogger(__name__)


class ApiError(RuntimeError):
    """An API call failed in a way retrying will not fix."""

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class OffsetLimitReached(ApiError):
    """Gamma refuses offsets past its cap. Not a failure: stop and use what you have."""


class Client:
    def __init__(self, cfg: ApiConfig | None = None):
        self.cfg = cfg or ApiConfig()
        self.session = requests.Session()
        self._calls = 0

    def get(self, url: str, params: dict | None = None) -> Any:
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=self.cfg.timeout_sec)
                self._calls += 1

                if r.status_code == 429:
                    wait = 2 ** attempt
                    log.warning("rate limited, backing off %ds", wait)
                    time.sleep(wait)
                    continue

                if 400 <= r.status_code < 500:
                    body = r.text[:300]
                    if r.status_code == 422 and "offset" in body.lower():
                        raise OffsetLimitReached(
                            "Gamma pagination limit reached", r.status_code, body)
                    hint = {
                        451: "blocked for legal reasons in this jurisdiction",
                        403: "rejected, often a bot rule",
                        404: "wrong path or the endpoint moved",
                    }.get(r.status_code, "")
                    raise ApiError(
                        f"HTTP {r.status_code} from {url}"
                        + (f" ({hint})" if hint else "")
                        + f"\n  body: {body}",
                        r.status_code, body)

                r.raise_for_status()
                time.sleep(self.cfg.sleep_sec)
                return r.json()

            except ApiError:
                raise
            except Exception as e:  # network-level; worth retrying
                last_err = e
                if attempt < self.cfg.max_retries - 1:
                    time.sleep(2 ** attempt)

        raise ApiError(
            f"failed after {self.cfg.max_retries} attempts: {url}\n"
            f"  last error: {type(last_err).__name__}: {str(last_err)[:200]}\n"
            f"  a connect timeout here usually means a network-level block, "
            f"not a code problem")

    def paginate(self, url: str, params: dict | None = None,
                 max_items: int | None = None, page_size: int | None = None):
        """Yield items across pages, stopping cleanly at the API's offset cap."""
        params = dict(params or {})
        size = page_size or self.cfg.page_size
        offset, seen = 0, 0
        while True:
            if max_items is not None and seen >= max_items:
                return
            if offset > self.cfg.max_offset:
                log.info("reached configured max_offset=%d", self.cfg.max_offset)
                return
            take = size if max_items is None else min(size, max_items - seen)
            try:
                batch = self.get(url, {**params, "limit": take, "offset": offset})
            except OffsetLimitReached:
                log.info("API pagination limit at offset %d; using what we have", offset)
                return
            if not batch:
                return
            for item in batch:
                yield item
            seen += len(batch)
            if len(batch) < take:
                return
            offset += len(batch)

    @property
    def call_count(self) -> int:
        return self._calls
