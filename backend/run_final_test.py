"""Run the precommitted final market comparison on the held-out test window.

This script deliberately has no tuning logic. It evaluates three variants and
five fixed seeds once, then writes a compact CSV. The test window must not be
used to change these choices afterwards.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace

import pandas as pd

from polymap.config import load_config
from polymap.evaluation.baseline_eval import evaluate_scores, score_baselines
from polymap.pipeline import _candidate_bundle, prepare_model_bundle
from polymap.training.negatives import build_sampler
from polymap.training.splits import temporal_split
from polymap.training.trainer import score_events, train

VARIANTS = {
    # Leak-free plain model, with the optimization settings used for the tuned
    # comparison but without the pair-level residual head.
    "plain": {"pair_features": False, "memory_updates": True,
               "epochs": 10, "learning_rate": 0.003},
    "tuned": {"pair_features": True, "memory_updates": True,
              "epochs": 20, "learning_rate": 0.003},
    # Pair history only ablation: static node representations plus the causal
    # pair head, using the untouched default learning rate and 10 epochs.
    "memory_off": {"pair_features": True, "memory_updates": False,
                   "epochs": 10, "learning_rate": 0.001},
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", action="append", default=[])
    p.add_argument("--run-name", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--output", help="CSV path for per-seed results")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(*args.config, run_name=args.run_name)
    base = _candidate_bundle(cfg)
    split = temporal_split(base, cfg.split.train_fraction)
    rows = []

    for variant, settings in VARIANTS.items():
        for seed in args.seeds:
            model_cfg = replace(
                cfg.model, seed=seed, pair_features=settings["pair_features"],
                memory_updates=settings["memory_updates"],
                train_only_features=True, epochs=settings["epochs"],
                learning_rate=settings["learning_rate"], device="cpu")
            run_cfg = replace(cfg, model=model_cfg)
            model_bundle = prepare_model_bundle(run_cfg, base, split.train_idx)
            result = train(model_bundle, split, model_cfg, negative="hard")
            sampler = build_sampler("hard", model_bundle.edge_index,
                                    split.train_idx, model_bundle.num_nodes)
            scored = score_events(
                result, model_bundle, split, sampler, seed=seed,
                n_negatives=model_cfg.eval_negatives,
                batch_size=model_cfg.eval_batch_size)
            gap = scored["events"]["gap_s"].to_numpy()
            metrics = evaluate_scores(
                scored["positive"], scored["negative"], gap, cfg.graph.window_seconds)
            baseline_scores = score_baselines(
                model_bundle, split, scored["neg_ids"], model_cfg.eval_batch_size,
                names=("recency",))
            baseline = evaluate_scores(
                baseline_scores["recency"]["positive"],
                baseline_scores["recency"]["negative"], gap, cfg.graph.window_seconds)
            rows.append({
                "variant": variant,
                "seed": seed,
                "train_events": len(split.train_idx),
                "test_events": len(split.test_idx),
                "mrr": metrics["mrr"],
                "roc_auc": metrics["roc_auc"],
                "hits@10": metrics["hits@10"],
                "recency_mrr_same_negatives": baseline["mrr"],
                "mrr_gap_vs_recency": metrics["mrr"] - baseline["mrr"],
            })
            print(f"{variant:11s} seed={seed}: MRR={metrics['mrr']:.3f} "
                  f"AUC={metrics['roc_auc']:.3f} "
                  f"recency={baseline['mrr']:.3f}")

    results = pd.DataFrame(rows)
    summary = (results.groupby("variant", as_index=False)
               .agg(mrr_mean=("mrr", "mean"), mrr_std=("mrr", "std"),
                    auc_mean=("roc_auc", "mean"),
                    recency_mrr_mean=("recency_mrr_same_negatives", "mean"),
                    gap_vs_recency_mean=("mrr_gap_vs_recency", "mean"),
                    seeds=("seed", "count")))
    output = args.output or str(cfg.artifacts_dir / "final_test_results.csv")
    results.to_csv(output, index=False)
    summary_path = output.removesuffix(".csv") + "_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nsummary:")
    print(summary.to_string(index=False))
    print(f"\nper-seed: {output}\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
