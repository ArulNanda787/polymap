"""Run model variants and baselines on one fixed protocol.

Used by `run_benchmark.py`, `run_ablations.py` and `prospective.py`, so all three
produce numbers that are comparable by construction: same split, same fixed
negative sets, same timestamp-grouped batches, same dedupe mask.

Feature-group ablation
----------------------
`PairHistory` emits 6 features: [0] log pair count, [1..3] decayed counts at
5 min / 1 h / 24 h, [4] seen flag, [5] log pair age. To remove a group WITHOUT
touching the trainer, `pair_feature_mask` patches `PolymapTGN.__init__` for the
duration of one `train()` call and multiplies the pair-head input by a 0/1 mask.
The head then sees constant zeros for those columns, which is equivalent to not
having the feature (the bias absorbs the constant). Training budget, optimiser
and data are untouched, which is the point of a controlled ablation.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..config import Config
from ..graph.build import GraphBundle
from ..training.splits import TemporalSplit
from .protocol import (NegativeSet, block_bootstrap, dedupe_mask, pair_gap,
                       score_baselines_strict, score_events_strict, summarize_scores)

log = logging.getLogger(__name__)

PAIR_FEATURE_GROUPS = {"count": [0], "decayed": [1, 2, 3], "seen": [4], "age": [5]}
DROP_ALIASES = {"recency": ("decayed", "age"), "count_and_seen": ("count", "seen")}


def feature_mask(drop: tuple[str, ...]) -> np.ndarray:
    m = np.ones(6, dtype=np.float32)
    for g in drop:
        for name in DROP_ALIASES.get(g, (g,)):
            for i in PAIR_FEATURE_GROUPS[name]:
                m[i] = 0.0
    return m


@contextmanager
def pair_feature_mask(mask: np.ndarray):
    import torch

    from ..models.temporal import PolymapTGN

    original = PolymapTGN.__init__
    mask_t = torch.tensor(mask, dtype=torch.float32)

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.pair_head is not None:
            self.pair_head.register_forward_pre_hook(
                lambda module, inputs: (inputs[0] * mask_t.to(inputs[0].device),))

    PolymapTGN.__init__ = patched
    try:
        yield
    finally:
        PolymapTGN.__init__ = original


@dataclass(frozen=True)
class VariantSpec:
    name: str
    pair_features: bool = True
    memory_updates: bool = True
    drop_groups: tuple[str, ...] = ()
    train_negative: str = "hard"
    epochs: int = 10
    learning_rate: float = 0.003
    hidden_dim: int | None = None
    train_only_features: bool = True


def run_variant(cfg: Config, base_bundle: GraphBundle, split: TemporalSplit,
                spec: VariantSpec, seeds: list[int], eval_sets: dict[str, NegativeSet],
                *, dedupe: bool = True, max_batch: int | None = None,
                device: str | None = None, extra: dict | None = None):
    """Train `spec` once per seed; score every eval negative set.

    Returns (rows, rr) where rr[(spec.name, kind)] has shape (n_seeds, n_events).
    """
    from ..pipeline import prepare_model_bundle
    from ..training.trainer import train

    mask = dedupe_mask(base_bundle, split) if dedupe else np.ones(len(split.test_idx), bool)
    gap = pair_gap(base_bundle, split)
    rows: list[dict] = []
    rr_lists: dict[tuple[str, str], list[np.ndarray]] = {}

    for seed in seeds:
        mc = replace(cfg.model, seed=seed, pair_features=spec.pair_features,
                     memory_updates=spec.memory_updates,
                     train_only_features=spec.train_only_features,
                     epochs=spec.epochs, learning_rate=spec.learning_rate)
        if spec.hidden_dim:
            mc = replace(mc, hidden_dim=spec.hidden_dim)
        if device:
            mc = replace(mc, device=device)
        run_cfg = replace(cfg, model=mc)
        model_bundle = prepare_model_bundle(run_cfg, base_bundle, split.train_idx)
        ctx = (pair_feature_mask(feature_mask(spec.drop_groups))
               if (spec.drop_groups and spec.pair_features) else nullcontext())
        with ctx:
            result = train(model_bundle, split, mc, negative=spec.train_negative)
        for kind, ns in eval_sets.items():
            pos, neg = score_events_strict(result, model_bundle, split, ns.ids,
                                           max_batch or mc.eval_batch_size)
            metrics, rr = summarize_scores(pos, neg, gap, cfg.graph.window_seconds, mask)
            rows.append({"model": spec.name, "seed": seed, "eval_negative": kind,
                         "train_negative": spec.train_negative, **metrics, **(extra or {})})
            rr_lists.setdefault((spec.name, kind), []).append(rr)
            print(f"    {spec.name} seed={seed} [{kind}] MRR {metrics['mrr']:.3f} "
                  f"AUC {metrics['roc_auc']:.3f}", flush=True)
    return rows, {k: np.vstack(v) for k, v in rr_lists.items()}


def run_baselines(cfg: Config, base_bundle: GraphBundle, split: TemporalSplit,
                  names: tuple[str, ...], eval_sets: dict[str, NegativeSet],
                  *, dedupe: bool = True, max_batch: int | None = None,
                  extra: dict | None = None):
    mask = dedupe_mask(base_bundle, split) if dedupe else np.ones(len(split.test_idx), bool)
    gap = pair_gap(base_bundle, split)
    rows, rr_out = [], {}
    batch = max_batch or cfg.model.eval_batch_size
    for kind, ns in eval_sets.items():
        scores = score_baselines_strict(base_bundle, split, ns.ids, batch, tuple(names))
        for name, sc in scores.items():
            metrics, rr = summarize_scores(sc["positive"], sc["negative"], gap,
                                           cfg.graph.window_seconds, mask)
            rows.append({"model": name, "seed": -1, "eval_negative": kind,
                         "train_negative": "n/a", **metrics, **(extra or {})})
            rr_out[(name, kind)] = rr[None, :]
    return rows, rr_out


def summarize(rows: list[dict] | pd.DataFrame, group=("model", "eval_negative")) -> pd.DataFrame:
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    agg = df.groupby(list(group), as_index=False).agg(
        n_events=("n_events", "first"), n_seeds=("seed", "nunique"),
        mrr_mean=("mrr", "mean"), mrr_std=("mrr", "std"),
        mrr_min=("mrr", "min"), mrr_max=("mrr", "max"),
        hits10_mean=("hits@10", "mean"), hits10_std=("hits@10", "std"),
        auc_mean=("roc_auc", "mean"), auc_std=("roc_auc", "std"))
    return agg


def paired(rr: dict, kind: str, a: str, b: str, seed: int = 0, **boot) -> dict:
    """MRR(a) - MRR(b) on identical events. Seed-averaged per-event RR, block bootstrap CI."""
    d = rr[(a, kind)].mean(axis=0) - rr[(b, kind)].mean(axis=0)
    return {"a": a, "b": b, "eval_negative": kind, **block_bootstrap(d, seed=seed, **boot)}


def paired_table(rr: dict, kinds, a_models, b_models, **boot) -> pd.DataFrame:
    out = []
    for kind in kinds:
        for a in a_models:
            for b in b_models:
                if a != b and (a, kind) in rr and (b, kind) in rr:
                    r = paired(rr, kind, a, b, **boot)
                    r["ci_excludes_zero"] = bool(r["ci95_low"] > 0 or r["ci95_high"] < 0)
                    out.append(r)
    return pd.DataFrame(out)
