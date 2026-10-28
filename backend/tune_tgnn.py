"""Tune Polymap's model on a validation window without touching the test window.

Example:

    python tune_tgnn.py -c configs/default.yaml \
      -c configs/markets/xi_2027.yaml --candidates-only --negative hard \
      --learning-rates 0.001 0.003 --epochs 10 20 --seeds 42 43 44 \
      --set model.pair_features=true --set model.train_only_features=true \
      --set split.validation_split=0.15

The selected row is only a validation result. Run the final test configuration
separately after choosing hyperparameters; this script never scores test events.
"""
from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import replace

import pandas as pd

from polymap.config import load_config, parse_set_expression
from polymap.evaluation.baseline_eval import evaluate_scores
from polymap.graph.build import build_graph
from polymap.pipeline import _candidate_bundle, prepare_model_bundle
from polymap.training.negatives import build_sampler
from polymap.training.splits import temporal_split, validation_view
from polymap.training.trainer import score_events, train

log = logging.getLogger(__name__)


def _config(paths, expressions, run_name=None):
    overrides = {}
    for expression in expressions:
        section, field_name, value = parse_set_expression(expression)
        overrides.setdefault(section, {})[field_name] = value
    if run_name:
        overrides["run_name"] = run_name
    return load_config(*paths, **overrides)


def _bundle(cfg, candidates_only):
    if candidates_only:
        return _candidate_bundle(cfg)
    edges = pd.read_parquet(cfg.processed_dir / "temporal_edges.parquet")
    features = pd.read_parquet(cfg.processed_dir / "wallet_features.parquet")
    return build_graph(edges, features)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", action="append", default=[])
    p.add_argument("--run-name", help="use another processed/artifact run directory")
    p.add_argument("--set", dest="set_values", action="append", default=[],
                   metavar="SECTION.FIELD=VALUE")
    p.add_argument("--negative", choices=["random", "hard", "popularity"], default="hard")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--learning-rates", type=float, nargs="+", default=[0.001, 0.003])
    p.add_argument("--epochs", type=int, nargs="+", default=[10, 20])
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[32])
    p.add_argument("--candidates-only", action="store_true")
    p.add_argument("--output", help="CSV path (default: data/artifacts/<run>/tuning_results.csv)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = _config(args.config, args.set_values, args.run_name)
    if not 0.0 < cfg.split.validation_split < 1.0:
        p.error("tuning requires --set split.validation_split=<fraction> between 0 and 1")

    base = _bundle(cfg, args.candidates_only)
    split = temporal_split(base, cfg.split.train_fraction, cfg.split.validation_split)
    fit = validation_view(split)
    log.info("validation tuning: fit=%d validation=%d; test is not scored",
             len(fit.train_idx), len(fit.test_idx))

    rows = []
    grid = itertools.product(args.learning_rates, args.epochs, args.hidden_dims, args.seeds)
    for learning_rate, epochs, hidden_dim, seed in grid:
        model_cfg = replace(cfg.model, learning_rate=learning_rate, epochs=epochs,
                            hidden_dim=hidden_dim, seed=seed)
        run_cfg = replace(cfg, model=model_cfg)
        model_bundle = prepare_model_bundle(run_cfg, base, fit.train_idx)
        result = train(model_bundle, fit, model_cfg, negative=args.negative)
        sampler = build_sampler(args.negative, model_bundle.edge_index,
                                fit.train_idx, model_bundle.num_nodes)
        scored = score_events(
            result, model_bundle, fit, sampler, seed=seed,
            n_negatives=model_cfg.eval_negatives, batch_size=model_cfg.eval_batch_size)
        metrics = evaluate_scores(
            scored["positive"], scored["negative"],
            scored["events"]["gap_s"].to_numpy(), cfg.graph.window_seconds)
        rows.append({
            "learning_rate": learning_rate,
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "seed": seed,
            "negative": args.negative,
            "pair_features": model_cfg.pair_features,
            "train_only_features": model_cfg.train_only_features,
            "validation_events": len(fit.test_idx),
            "validation_mrr": metrics["mrr"],
            "validation_roc_auc": metrics["roc_auc"],
            "validation_hits@10": metrics["hits@10"],
        })
        log.info("lr=%g epochs=%d hidden=%d seed=%d -> validation MRR %.3f",
                 learning_rate, epochs, hidden_dim, seed, metrics["mrr"])

    results = pd.DataFrame(rows)
    summary = (results.groupby(["learning_rate", "epochs", "hidden_dim"], as_index=False)
               .agg(validation_mrr_mean=("validation_mrr", "mean"),
                    validation_mrr_std=("validation_mrr", "std"),
                    validation_auc_mean=("validation_roc_auc", "mean"),
                    seeds=("seed", "count"))
               .sort_values("validation_mrr_mean", ascending=False))
    out = args.output or str(cfg.artifacts_dir / "tuning_results.csv")
    summary_path = out.removesuffix(".csv") + "_summary.csv"
    results.to_csv(out, index=False)
    summary.to_csv(summary_path, index=False)
    print(results.to_string(index=False))
    print("\nvalidation ranking:")
    print(summary.to_string(index=False))
    print(f"\nraw results: {out}\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
