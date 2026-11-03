"""Controlled ablations: change ONE factor, hold everything else fixed.

Fixed across every row of every table: training budget (epochs, learning rate,
batch size), seeds, chronological split, fixed negative sets, timestamp-grouped
scoring, same-timestamp dedupe. The only thing that moves is the named factor.

Blocks (choose with --blocks, default all):

  components   full | no_memory | no_pair_history | no_memory_no_pair_history |
               no_recency_features | no_count_features
  neg_train    train negative sampler: random | hard | popularity
               (evaluated against every eval negative set)
  window       trading-time window W: edges with time_diff <= W
  min_events   minimum interactions per pair for the candidate universe

Notes on interpretation
-----------------------
* components / neg_train use the SAME events for every variant, so differences
  are paired and get block-bootstrap CIs against `full`.
* window / min_events change WHICH events exist (or which wallets are in the
  universe), so absolute MRR is not comparable across rows. The comparable
  quantity is the model's gap to the recency baseline on the SAME events, which
  is reported in `gap_vs_recency_*` columns.
* window uses the exact edge table built at the config's window (default 300 s)
  filtered to time_diff <= W. With max_pairs_per_trade this is identical to
  rebuilding at W, because the cap keeps the earliest following trades.
* min_events re-runs the model-free candidate screen on the FULL observation
  period (as the original study did). It is a sensitivity analysis, not a
  leak-free universe; `prospective.py` builds the leak-free universe.

Example:
    python run_ablations.py -c configs/default.yaml -c configs/markets/xi_2027.yaml \
        --run-name xi_2027_63d_p50 --seeds 42 43 44 45 46 --epochs 10 --lr 0.003 \
        --eval-negatives hard active
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from polymap.analysis.relationships import candidate_wallets, prescreen_pairs, select_candidates
from polymap.config import load_config
from polymap.evaluation.experiment import (VariantSpec, paired_table, run_baselines,
                                           run_variant, summarize)
from polymap.evaluation.protocol import build_eval_negatives
from polymap.graph.build import build_graph
from polymap.models import baselines_extra  # noqa: F401
from polymap.pipeline import _candidate_bundle
from polymap.training.splits import temporal_split


def _bundle_from_edges(edges: pd.DataFrame, features: pd.DataFrame):
    return build_graph(edges, features)


def _gap_rows(rows: list[dict], rr: dict, model: str, ref: str, kinds, tag: dict) -> list[dict]:
    """Model-minus-reference MRR on identical events, for sweeps where events change."""
    from polymap.evaluation.protocol import block_bootstrap
    out = []
    for k in kinds:
        d = rr[(model, k)].mean(axis=0) - rr[(ref, k)].mean(axis=0)
        out.append({**tag, "eval_negative": k, "model": model, "reference": ref,
                    **{f"gap_vs_{ref}_{key}": v for key, v in block_bootstrap(d).items()
                       if key in ("mean", "ci95_low", "ci95_high")}})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", action="append", default=[])
    p.add_argument("--run-name", required=True)
    p.add_argument("--blocks", nargs="+", default=["components", "neg_train", "window", "min_events"],
                   choices=["components", "neg_train", "window", "min_events"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--epochs", type=int, default=10, help="FIXED for every variant")
    p.add_argument("--lr", type=float, default=0.003, help="FIXED for every variant")
    p.add_argument("--eval-negatives", nargs="+", default=["hard", "active"],
                   choices=["random", "popularity", "hard", "active", "inductive"])
    p.add_argument("--windows", type=int, nargs="+", default=[30, 60, 120, 300])
    p.add_argument("--min-events-list", type=int, nargs="+", default=[2, 3, 5, 8])
    p.add_argument("--n-negatives", type=int, default=50)
    p.add_argument("--eval-seed", type=int, default=12345)
    p.add_argument("--no-dedupe", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(*a.config, run_name=a.run_name)
    out = Path(a.outdir) if a.outdir else cfg.artifacts_dir / "ablations"
    out.mkdir(parents=True, exist_ok=True)
    dedupe = not a.no_dedupe
    kinds = tuple(a.eval_negatives)

    def spec(name, **kw):
        return VariantSpec(name=name, epochs=a.epochs, learning_rate=a.lr, **kw)

    def make_eval(bundle, split):
        return {k: build_eval_negatives(bundle, split, k, a.n_negatives, a.eval_seed) for k in kinds}

    # ------------------------------------------------------------ components
    if "components" in a.blocks or "neg_train" in a.blocks:
        base = _candidate_bundle(cfg)
        split = temporal_split(base, cfg.split.train_fraction)
        eval_sets = make_eval(base, split)
        print(f"events: {len(split.train_idx)} train / {len(split.test_idx)} test")

    if "components" in a.blocks:
        specs = [
            spec("full"),
            spec("no_memory", memory_updates=False),
            spec("no_pair_history", pair_features=False),
            spec("no_memory_no_pair_history", pair_features=False, memory_updates=False),
            spec("no_recency_features", drop_groups=("recency",)),
            spec("no_count_features", drop_groups=("count_and_seen",)),
        ]
        rows, rr = run_baselines(cfg, base, split, ("recency", "edgebank_unlimited"),
                                 eval_sets, dedupe=dedupe, extra={"block": "components"})
        for s in specs:
            r, rr_s = run_variant(cfg, base, split, s, a.seeds, eval_sets, dedupe=dedupe,
                                  device=a.device, extra={"block": "components"})
            rows += r
            rr.update(rr_s)
        df = pd.DataFrame(rows)
        df.to_csv(out / "components_per_seed.csv", index=False)
        summarize(df).to_csv(out / "components_summary.csv", index=False)
        paired_table(rr, kinds, [s.name for s in specs if s.name != "full"], ["full"]).to_csv(
            out / "components_paired_vs_full.csv", index=False)
        paired_table(rr, kinds, [s.name for s in specs], ["recency"]).to_csv(
            out / "components_paired_vs_recency.csv", index=False)
        print("\n[components] mean over seeds")
        print(summarize(df).round(3).to_string(index=False))

    # ------------------------------------------------------- training negatives
    if "neg_train" in a.blocks:
        rows, rr = [], {}
        specs = [spec(f"train_{t}", train_negative=t) for t in ("random", "hard", "popularity")]
        for s in specs:
            r, rr_s = run_variant(cfg, base, split, s, a.seeds, eval_sets, dedupe=dedupe,
                                  device=a.device, extra={"block": "neg_train"})
            rows += r
            rr.update(rr_s)
        df = pd.DataFrame(rows)
        df.to_csv(out / "neg_train_per_seed.csv", index=False)
        summarize(df).to_csv(out / "neg_train_summary.csv", index=False)
        paired_table(rr, kinds, ["train_random", "train_popularity"], ["train_hard"]).to_csv(
            out / "neg_train_paired_vs_hard.csv", index=False)
        print("\n[neg_train] mean over seeds")
        print(summarize(df).round(3).to_string(index=False))

    # ----------------------------------------------------- window / min events
    def sweep(label: str, items, build):
        rows, gaps = [], []
        for item in items:
            built = build(item)
            if built is None:
                continue
            edges, features = built
            bundle = _bundle_from_edges(edges, features)
            sp = temporal_split(bundle, cfg.split.train_fraction)
            es = make_eval(bundle, sp)
            tag = {"block": label, label: item, "n_events_total": bundle.num_edges,
                   "n_wallets": bundle.num_nodes}
            r, rr = run_baselines(cfg, bundle, sp, ("recency",), es, dedupe=dedupe, extra=tag)
            r2, rr2 = run_variant(cfg, bundle, sp, spec("full"), a.seeds, es,
                                  dedupe=dedupe, device=a.device, extra=tag)
            rows += r + r2
            rr.update(rr2)
            gaps += _gap_rows(rows, rr, "full", "recency", kinds, tag)
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out / f"{label}_per_seed.csv", index=False)
            pd.DataFrame(gaps).to_csv(out / f"{label}_gap_vs_recency.csv", index=False)
            print(f"\n[{label}] model gap to recency on identical events")
            print(pd.DataFrame(gaps).round(3).to_string(index=False))

    if "window" in a.blocks or "min_events" in a.blocks:
        edges_all = pd.read_parquet(cfg.processed_dir / "temporal_edges.parquet")
        feats_all = pd.read_parquet(cfg.processed_dir / "wallet_features.parquet")
        universe = set((cfg.processed_dir / "candidate_wallets.txt").read_text().split())
        in_u = edges_all.source.isin(universe) & edges_all.target.isin(universe)

    if "window" in a.blocks:
        wmax = cfg.graph.window_seconds

        def build_window(w):
            if w > wmax:
                print(f"skip window {w}: the built edge table only reaches {wmax}s")
                return None
            e = edges_all[in_u & (edges_all.time_diff <= w)]
            return (e, feats_all) if len(e) > 100 else None

        sweep("window", [w for w in a.windows], build_window)

    if "min_events" in a.blocks:
        def build_min_events(m):
            scr = replace(cfg.screening, min_events=m)
            chosen = select_candidates(prescreen_pairs(edges_all, scr))
            wallets = set(candidate_wallets(chosen))
            e = edges_all[edges_all.source.isin(wallets) & edges_all.target.isin(wallets)]
            print(f"min_events={m}: {len(chosen)} pairs, {len(wallets)} wallets, {len(e)} edges")
            return (e, feats_all) if len(e) > 100 else None

        sweep("min_events", a.min_events_list, build_min_events)

    print(f"\nwrote tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
