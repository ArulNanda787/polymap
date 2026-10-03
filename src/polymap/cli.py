"""Command line entry point: `polymap <stage>`."""
from __future__ import annotations

import argparse
import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="polymap", description="Polymap pipeline")
    p.add_argument("stage", choices=["fetch", "build", "train", "screen", "all", "markets"])
    p.add_argument("-c", "--config", action="append", default=[],
                   help="config file; repeat to layer them, later wins")
    p.add_argument("--run-name", help="names the output directory under data/")
    p.add_argument("--markets", nargs="+", help="override the market list")
    p.add_argument("--window", type=int, help="override graph.window_seconds")
    p.add_argument("--negative", default="random",
                   choices=["random", "hard", "popularity"])
    p.add_argument("--refresh", action="store_true",
                   help="refetch and REPLACE the cached slice (changes all results)")
    p.add_argument("--min-volume", type=float, default=100_000, help="for `markets`")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    setup_logging(a.verbose)
    from .config import load_config
    from . import pipeline

    overrides = {}
    if a.run_name:
        overrides["run_name"] = a.run_name
    if a.markets:
        overrides["markets"] = a.markets
    if a.window:
        overrides["graph"] = {"window_seconds": a.window}

    cfg = load_config(*a.config, **overrides)
    log = logging.getLogger("polymap")
    log.info("run=%s fingerprint=%s markets=%s",
             cfg.run_name, cfg.fingerprint(), cfg.markets or "(none)")

    try:
        if a.stage == "markets":
            from .data.trades import discover_markets
            df = discover_markets(cfg, min_volume=a.min_volume, closed=True, limit=40)
            if df.empty:
                print("no markets matched")
            else:
                print(df[["market_id", "slug", "volume", "resolved"]].to_string(index=False))
        elif a.stage == "fetch":
            pipeline.stage_fetch(cfg, refresh=a.refresh)
        elif a.stage == "build":
            pipeline.stage_build(cfg)
        elif a.stage == "train":
            out = pipeline.stage_train(cfg, negative=a.negative)
            print(__import__("json").dumps(out["metrics"], indent=2))
        elif a.stage == "screen":
            df = pipeline.stage_screen(cfg)
            print(df["category"].value_counts().to_string())
        elif a.stage == "all":
            pipeline.run_all(cfg, refresh=a.refresh, negative=a.negative)
    except Exception as e:
        log.error("%s: %s", type(e).__name__, e)
        if a.verbose:
            raise
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
