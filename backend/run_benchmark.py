"""Fair multi-market benchmark: your model vs. simple baselines vs. (optionally) TGN.

Everything shares, per market: one chronological split, one candidate-universe
graph, ONE fixed negative set per sampler (independent of model seeds), the
same timestamp-grouped batches and the same same-timestamp dedupe. Differences
therefore come from the method, not the protocol.

Example (run from backend/, after build + candidates for each market):

    python run_benchmark.py -c configs/default.yaml \
        --run-names xi_2027_63d_p50 rep_iran iran_2027 \
        --samplers hard active popularity random \
        --variants tuned plain memory_off \
        --seeds 42 43 44 45 46 --tgn

Outputs (default data/artifacts/benchmark/):
    per_seed.csv        one row per (market, model, seed, eval negative)
    summary.csv         mean/std/min/max across seeds
    paired.csv          paired MRR differences + block-bootstrap 95% CI
    cross_market.csv    one line per (variant, sampler, comparator): mean gap and #markets with CI > 0
    protocol.json       negative-set diagnostics (collision rates, fallbacks), event counts

Paired differences use the SEED-AVERAGED per-event reciprocal rank of the model
(an estimate of its expected performance) against each baseline's per-event
reciprocal rank on identical events, resampled in time blocks.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from polymap.config import load_config
from polymap.evaluation.experiment import (VariantSpec, paired_table, run_baselines,
                                           run_variant, summarize)
from polymap.evaluation.protocol import (build_eval_negatives, dedupe_mask, pair_gap,
                                         summarize_scores)
from polymap.models import baselines_extra  # noqa: F401  (registers EdgeBank, Adamic-Adar)
from polymap.pipeline import _candidate_bundle, prepare_model_bundle
from polymap.training.splits import temporal_split
from run_final_test import VARIANTS  # the precommitted variant definitions

DEFAULT_BASELINES = ("recency", "pair_count", "common_neighbours", "adamic_adar",
                     "popularity", "edgebank_unlimited", "edgebank_24h", "edgebank_1h")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", action="append", default=[])
    p.add_argument("--run-names", nargs="+", required=True)
    p.add_argument("--samplers", nargs="+", default=["hard", "active", "popularity", "random"],
                   choices=["random", "popularity", "hard", "active", "inductive"])
    p.add_argument("--variants", nargs="+", default=["tuned"], choices=list(VARIANTS))
    p.add_argument("--baselines", nargs="+", default=list(DEFAULT_BASELINES))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--n-negatives", type=int, default=50)
    p.add_argument("--eval-seed", type=int, default=12345,
                   help="seed of the fixed negative sets; independent of model seeds")
    p.add_argument("--active-window", type=int, default=3600)
    p.add_argument("--train-negative", default="hard", choices=["random", "hard", "popularity"])
    p.add_argument("--no-dedupe", action="store_true",
                   help="keep repeated rows of the same pair at the same timestamp")
    p.add_argument("--tgn", action="store_true", help="also train/evaluate TGN (needs torch_geometric)")
    p.add_argument("--tgn-epochs", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs-override", type=int,
                   help="SMOKE TEST ONLY: force every variant to this many epochs")
    p.add_argument("--outdir", default="data/artifacts/benchmark")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dedupe = not a.no_dedupe

    all_rows, all_paired, protocol = [], [], {}
    for rn in a.run_names:
        cfg = load_config(*a.config, run_name=rn)
        base = _candidate_bundle(cfg)
        split = temporal_split(base, cfg.split.train_fraction)
        eval_sets = {k: build_eval_negatives(base, split, k, a.n_negatives, a.eval_seed,
                                             a.active_window) for k in a.samplers}
        protocol[rn] = {"train_events": int(len(split.train_idx)),
                        "test_events": int(len(split.test_idx)),
                        "test_events_after_dedupe": int(dedupe_mask(base, split).sum()),
                        "cold_start_events": split.n_cold_start,
                        "negatives": {k: v.meta for k, v in eval_sets.items()}}
        print(f"\n=== {rn}: {protocol[rn]['train_events']} train / "
              f"{protocol[rn]['test_events']} test events "
              f"({protocol[rn]['test_events_after_dedupe']} after same-timestamp dedupe)")
        extra = {"market_run": rn}

        rows, rr = run_baselines(cfg, base, split, tuple(a.baselines), eval_sets,
                                 dedupe=dedupe, extra=extra)
        for v in a.variants:
            s = VARIANTS[v]
            spec = VariantSpec(name=v, pair_features=s["pair_features"],
                               memory_updates=s["memory_updates"],
                               epochs=a.epochs_override or s["epochs"],
                               learning_rate=s["learning_rate"], train_negative=a.train_negative)
            r, rr_v = run_variant(cfg, base, split, spec, a.seeds, eval_sets,
                                  dedupe=dedupe, device=a.device, extra=extra)
            rows += r
            rr.update(rr_v)
            print(f"  variant {v} done", flush=True)

        if a.tgn:
            from polymap.models.tgn_pyg import run_tgn
            mask = dedupe_mask(base, split) if dedupe else np.ones(len(split.test_idx), bool)
            gap = pair_gap(base, split)
            tgn_rr: dict = {}
            mb = prepare_model_bundle(
                load_config(*a.config, run_name=rn, model={"train_only_features": True}),
                base, split.train_idx)
            for seed in a.seeds:
                scored = run_tgn(mb, split, {k: ns.ids for k, ns in eval_sets.items()},
                                 seed=seed, epochs=a.tgn_epochs,
                                 train_negative=a.train_negative, device=a.device)
                for kind, (pos, neg) in scored.items():
                    m, r1 = summarize_scores(pos, neg, gap, cfg.graph.window_seconds, mask)
                    rows.append({"model": "tgn", "seed": seed, "eval_negative": kind,
                                 "train_negative": a.train_negative, **m, **extra})
                    tgn_rr.setdefault(("tgn", kind), []).append(r1)
                print(f"  tgn seed {seed} done", flush=True)
            rr.update({k: np.vstack(v) for k, v in tgn_rr.items()})

        model_names = list(a.variants) + (["tgn"] if a.tgn else [])
        pt = paired_table(rr, a.samplers, model_names, list(a.baselines) + [m for m in model_names])
        pt.insert(0, "market_run", rn)
        all_rows += rows
        all_paired.append(pt)

        s = summarize(pd.DataFrame(rows))
        for kind in a.samplers:
            print(f"\n[{rn}] negatives = {kind}")
            print(s[s.eval_negative == kind].drop(columns="eval_negative")
                  .sort_values("mrr_mean", ascending=False).round(3).to_string(index=False))

    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    summ = summarize(per_seed, group=("market_run", "model", "eval_negative"))
    summ.to_csv(out / "summary.csv", index=False)
    paired_df = pd.concat(all_paired, ignore_index=True)
    paired_df.to_csv(out / "paired.csv", index=False)

    # cross-market: how often, and by how much, does each variant beat each comparator?
    cm = (paired_df[paired_df.a.isin(list(a.variants) + (["tgn"] if a.tgn else []))]
          .assign(wins=lambda d: d.ci95_low > 0, losses=lambda d: d.ci95_high < 0)
          .groupby(["a", "b", "eval_negative"], as_index=False)
          .agg(n_markets=("market_run", "nunique"), mean_mrr_gap=("mean", "mean"),
               min_mrr_gap=("mean", "min"), markets_ci_above_0=("wins", "sum"),
               markets_ci_below_0=("losses", "sum")))
    cm.to_csv(out / "cross_market.csv", index=False)
    (out / "protocol.json").write_text(json.dumps(
        {"dedupe_same_timestamp": dedupe, "n_negatives": a.n_negatives,
         "eval_seed": a.eval_seed, "seeds": a.seeds, "markets": protocol}, indent=2))
    print(f"\nwrote {out}/per_seed.csv, summary.csv, paired.csv, cross_market.csv, protocol.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
