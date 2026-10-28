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
    p.add_argument("stage", choices=["fetch", "build", "train", "candidates", "screen", "null", "validate-funding", "funding-robustness", "all", "markets"])
    p.add_argument("-c", "--config", action="append", default=[],
                   help="config file; repeat to layer them, later wins")
    p.add_argument("--run-name", help="names the output directory under data/")
    p.add_argument("--markets", nargs="+", help="override the market list")
    p.add_argument("--window", type=int, help="override graph.window_seconds")
    p.add_argument("--negative", default="random",
                   choices=["random", "hard", "popularity"])
    p.add_argument("--categories", nargs="+",
                   choices=["persistent_tight", "one_off_tight", "persistent_loose", "one_off_loose"],
                   help="categories kept by `candidates` (default: the two tight ones)")
    p.add_argument("--min-events", type=int,
                   help="override screening.min_events (pairs need at least this many edges)")
    p.add_argument("--max-pairs", type=int, default=0,
                   help="`candidates`: keep only the top N pairs (0 = all)")
    p.add_argument("--candidates-only", action="store_true",
                   help="`train`: use only edges between candidate wallets (run `candidates` first)")
    p.add_argument("--eval-samplers", nargs="+", choices=["random", "hard", "popularity"],
                   help="`train`: negative samplers to evaluate (default: all three)")
    p.add_argument("--baselines", nargs="*", metavar="NAME",
                    help="`train`: also score non-learned baselines on the same test events and "
                         "negatives. No names = all (pair_count recency popularity common_neighbours)")
    p.add_argument("--set", dest="set_values", action="append", metavar="SECTION.FIELD=VALUE",
                    help="override a config value, e.g. --set model.epochs=10; repeatable")
    p.add_argument("--funding-file", help="normalized on-chain transfers CSV/TSV/Parquet for validate-funding")
    p.add_argument("--candidates-file",
                   help="validate-funding: candidate pairs parquet (default: candidate_pairs.parquet "
                        "if present, else candidate_relationships.parquet)")
    p.add_argument("--exclude-wallets-file",
                   help="funding-robustness: csv (queried_wallet column) or txt of wallets to drop, "
                        "e.g. truncated_wallets.csv, as a sensitivity check")
    p.add_argument("--fanout-caps", type=int, nargs="+",
                   help="funding-robustness: hub caps to sweep (default 5 10 20 50 100)")
    p.add_argument("--controls-per-candidate", type=int, default=3, help="matched controls per TGNN candidate")
    p.add_argument("--max-funder-fanout", type=int, default=100,
                   help="exclude sources funding more than N analysis wallets from shared-funder evidence (0 disables)")
    p.add_argument("--funding-permutations", type=int, default=2000,
                   help="wallet-label permutations for direct-transfer test")
    p.add_argument("--n-null", type=int, default=20,
                   help="number of shuffled datasets per null mode (stage `null`)")
    p.add_argument("--null-mode",
                   choices=["default", "wallet", "stratified", "stratified_side",
                            "shift", "timestamp", "all"],
                   default="default",
                   help="null to run (stage `null`); `default` runs stratified + "
                        "stratified_side, `all` runs every mode")
    p.add_argument("--refresh", action="store_true",
                   help="refetch and REPLACE the cached slice (changes all results)")
    p.add_argument("--min-volume", type=float, default=100_000, help="for `markets`")
    p.add_argument("--open", action="store_true",
                   help="`markets`: list OPEN markets (default lists closed ones). Needed for "
                        "history_days fetches, which cover the last N days up to now.")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    setup_logging(a.verbose)
    from . import pipeline
    from .config import load_config, parse_set_expression

    overrides = {}
    if a.run_name:
        overrides["run_name"] = a.run_name
    if a.markets:
        overrides["markets"] = a.markets
    if a.window:
        overrides["graph"] = {"window_seconds": a.window}

    for expression in a.set_values or []:
        try:
            section, field_name, value = parse_set_expression(expression)
        except ValueError as e:
            p.error(str(e))
        overrides.setdefault(section, {})[field_name] = value

    if a.min_events:
        overrides["screening"] = {"min_events": a.min_events}

    cfg = load_config(*a.config, **overrides)
    log = logging.getLogger("polymap")
    log.info("run=%s fingerprint=%s markets=%s",
             cfg.run_name, cfg.fingerprint(), cfg.markets or "(none)")

    try:
        if a.stage == "markets":
            from .data.trades import discover_markets
            df = discover_markets(cfg, min_volume=a.min_volume, closed=not a.open, limit=40)
            if df.empty:
                print("no markets matched")
            else:
                print(df[["market_id", "slug", "volume", "resolved"]].to_string(index=False))
        elif a.stage == "fetch":
            pipeline.stage_fetch(cfg, refresh=a.refresh)
        elif a.stage == "build":
            pipeline.stage_build(cfg)
        elif a.stage == "train":
            baselines = None
            if a.baselines is not None:
                from .models.baselines import ALL_BASELINES
                baselines = tuple(a.baselines) or tuple(ALL_BASELINES)
                bad = [b for b in baselines if b not in ALL_BASELINES]
                if bad:
                    p.error(f"unknown baseline(s) {bad}; choose from {list(ALL_BASELINES)}")
            out = pipeline.stage_train(cfg, negative=a.negative,
                                       candidates_only=a.candidates_only,
                                       eval_samplers=tuple(a.eval_samplers) if a.eval_samplers else None,
                                       baselines=baselines)
            if baselines:
                import pandas as _pd
                tbl = _pd.read_csv(cfg.artifacts_dir / "baseline_comparison.csv")
                cols = [c for c in ("sampler", "model", "roc_auc", "mrr", "hits@10",
                                    "mrr_gap_tgnn_minus_this", "gap_ci95_low", "gap_ci95_high")
                        if c in tbl.columns]
                print(tbl[cols].round(3).to_string(index=False))
                print(f"\nfull table: {cfg.artifacts_dir / 'baseline_comparison.csv'}")
            else:
                print(__import__("json").dumps(out["metrics"], indent=2))
        elif a.stage == "candidates":
            df = pipeline.stage_candidates(cfg, categories=a.categories, max_pairs=a.max_pairs)
            print(df["category"].value_counts().to_string() if len(df) else "no candidates")
            print(f"wallets -> {cfg.processed_dir / 'candidate_wallets.txt'}")
        elif a.stage == "screen":
            df = pipeline.stage_screen(cfg)
            print(df["category"].value_counts().to_string())
        elif a.stage == "null":
            if a.null_mode == "all":
                from .analysis.null_model import MODES
                modes = MODES
            elif a.null_mode == "default":
                modes = None            # pipeline falls back to DEFAULT_MODES
            else:
                modes = (a.null_mode,)
            out = pipeline.stage_null(cfg, n_null=a.n_null, modes=modes)
            print(f"observed tight pairs: {out['observed']['n_tight']} "
                  f"(of {out['observed']['n_pairs']} pairs with enough edges)")
            for mode, r in out["nulls"].items():
                print(f"  null[{mode}]: {r['null_tight_mean']:.1f} +/- {r['null_tight_std']:.1f} "
                      f"(range {r['null_tight_min']}-{r['null_tight_max']})  p={r['empirical_p']:.3f}")
        elif a.stage == "validate-funding":
            if not a.funding_file:
                p.error("stage validate-funding requires --funding-file")
            out = pipeline.stage_validate_funding(
                cfg, a.funding_file, controls_per_candidate=a.controls_per_candidate,
                candidates_file=a.candidates_file,
                max_funder_fanout=a.max_funder_fanout,
                n_permutations=a.funding_permutations)
            print(__import__("json").dumps(out, indent=2))
        elif a.stage == "funding-robustness":
            if not a.funding_file:
                p.error("stage funding-robustness requires --funding-file")
            out = pipeline.stage_funding_robustness(
                cfg, a.funding_file, candidates_file=a.candidates_file,
                exclude_wallets_file=a.exclude_wallets_file, caps=a.fanout_caps)
            import pandas as _pd
            t = _pd.DataFrame(out["table"])
            print(f"candidate pairs used: {out['pairs']['candidate_pairs_used']} "
                  f"(dropped for no coverage: {out['pairs']['candidate_pairs_dropped_no_coverage']}); "
                  f"controls: {out['pairs']['control_pairs']}")
            show = t[["measure", "candidate", "control", "difference", "ci95_low", "ci95_high"]].copy()
            for c in show.columns[1:]:
                show[c] = show[c].map(lambda v: f"{v:+.3f}" if c in ("difference", "ci95_low", "ci95_high") else f"{v:.3f}")
            print(show.to_string(index=False))
            print(f"\nfull table: {cfg.artifacts_dir / 'funding_robustness_table.csv'}")
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
