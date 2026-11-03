"""One evaluation protocol shared by the benchmark, the ablations and the prospective test.

Why this exists
---------------
`trainer.score_events` draws its own negatives and batches test events by fixed
size. That is fine for development but has three problems for a paper:

1. **Negatives differ between methods.** Baselines reuse the TGNN's negatives
   (good), but nothing lets you hold ONE negative set fixed across seeds,
   variants and sampler comparisons. Here a `NegativeSet` is drawn once, from
   a seed that is independent of the model seed, and handed to every method.

2. **Negative collisions.** The old redraw only rejects the source and the true
   target. A "negative" that is another real partner of the source at the SAME
   timestamp is a true interaction scored as a miss. Every sampler here rejects
   (source, candidate, timestamp) triples that exist anywhere in the data.

3. **Simultaneous timestamps.** With fixed-size batches, two edges sharing a
   timestamp can land in different batches, so the second one is scored AFTER
   the first was absorbed into memory / pair history at age 0. That is future
   information about the same instant. `time_grouped_slices` never splits a
   timestamp group, so state seen by a scored event contains only events with
   a strictly earlier timestamp. `dedupe_mask` additionally keeps one test row
   per (undirected pair, timestamp), because repeated rows of the same pair at
   one second are one observation, not many.

Everything here needs only numpy/pandas/sklearn except `score_events_strict`,
which imports torch lazily.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..graph.build import GraphBundle
from ..training.negatives import HardNegativeSampler, PopularityNegativeSampler
from ..training.splits import TemporalSplit

log = logging.getLogger(__name__)

EVAL_KINDS = ("random", "popularity", "hard", "active", "inductive")


# ----------------------------------------------------------------- batching
def time_grouped_slices(ts: np.ndarray, max_events: int) -> list[slice]:
    """Consecutive slices of at most ~max_events rows that never split a timestamp.

    A group of equal timestamps larger than `max_events` becomes its own batch.
    Inside a batch state is stale (standard TGN approximation); across batches
    state contains only strictly earlier timestamps.
    """
    ts = np.asarray(ts)
    n = len(ts)
    if n == 0:
        return []
    change = np.flatnonzero(np.diff(ts) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])
    out: list[slice] = []
    cur = 0
    for g_start, g_end in zip(starts.tolist(), ends.tolist()):
        if g_end - cur > max_events and g_start > cur:
            out.append(slice(cur, g_start))
            cur = g_start
    out.append(slice(cur, n))
    return out


def dedupe_mask(bundle: GraphBundle, split: TemporalSplit) -> np.ndarray:
    """True for the first test row of each (undirected pair, timestamp)."""
    s, t = bundle.edge_index
    test = split.test_idx
    a = np.minimum(s[test], t[test]).astype(np.int64)
    b = np.maximum(s[test], t[test]).astype(np.int64)
    df = pd.DataFrame({"k": a * bundle.num_nodes + b, "ts": bundle.timestamps[test]})
    return ~df.duplicated(["k", "ts"]).to_numpy()


def pair_gap(bundle: GraphBundle, split: TemporalSplit) -> np.ndarray:
    """Seconds since the same undirected pair last interacted (NaN if never), for test events."""
    s, t = bundle.edge_index
    ts = bundle.timestamps
    a = np.minimum(s, t).astype(np.int64)
    b = np.maximum(s, t).astype(np.int64)
    prev = pd.Series(ts.astype(np.float64)).groupby(a * bundle.num_nodes + b).shift(1).to_numpy()
    return (ts - prev)[split.test_idx]


# ---------------------------------------------------------------- negatives
@dataclass
class NegativeSet:
    kind: str
    ids: np.ndarray            # (n_test, n_neg), aligned with split.test_idx
    seed: int
    meta: dict = field(default_factory=dict)


def build_eval_negatives(bundle: GraphBundle, split: TemporalSplit, kind: str,
                         n_neg: int = 50, seed: int = 12345,
                         active_window: int = 3600, max_redraws: int = 30) -> NegativeSet:
    """Draw one fixed, collision-free negative matrix for the test events.

    kinds
      random      uniform over all wallets.
      popularity  proportional to TRAIN degree (+1).
      hard        a wallet the source traded alongside in TRAIN (the existing
                  hard protocol). History-based baselines are disadvantaged here.
      active      a wallet that appears in any edge in the `active_window`
                  seconds BEFORE the event. Answers "who else was trading at
                  that moment?", which is what a recency/activity shortcut
                  cannot exploit. Causal: only strictly earlier edges are used.
      inductive   uniform, but never a wallet the source co-traded with in TRAIN.

    Rejected for every kind: the source itself, and any wallet that has an edge
    with the source at the same timestamp (in either direction).
    """
    if kind not in EVAL_KINDS:
        raise ValueError(f"unknown eval negative kind '{kind}', choose from {EVAL_KINDS}")
    rng = np.random.default_rng(seed)
    src, tgt = bundle.edge_index
    ts = bundle.timestamps
    n = bundle.num_nodes
    test = split.test_idx
    s = src[test].astype(np.int64)
    tt = ts[test]
    test_ts = np.unique(tt)
    K = len(test_ts)
    tidx = np.searchsorted(test_ts, tt)

    at_test_ts = np.isin(ts, test_ts)
    a = src[at_test_ts].astype(np.int64)
    b = tgt[at_test_ts].astype(np.int64)
    ti = np.searchsorted(test_ts, ts[at_test_ts])
    forbidden = np.unique(np.concatenate([(a * n + b) * K + ti, (b * n + a) * K + ti]))

    train_pairs = None
    if kind == "inductive":
        ta = src[split.train_idx].astype(np.int64)
        tb = tgt[split.train_idx].astype(np.int64)
        train_pairs = np.unique(np.concatenate([ta * n + tb, tb * n + ta]))

    if kind in ("random", "inductive"):
        def draw(srcs, times, k):
            return rng.integers(0, n, size=(len(srcs), k), dtype=np.int64)
    elif kind == "popularity":
        pop = PopularityNegativeSampler(bundle.edge_index, split.train_idx, n)

        def draw(srcs, times, k):
            return pop.sample_many(srcs, rng, k)
    elif kind == "hard":
        hard = HardNegativeSampler(bundle.edge_index, split.train_idx, n)

        def draw(srcs, times, k):
            return hard.sample_many(srcs, rng, k)
    else:  # active
        def draw(srcs, times, k):
            hi = np.searchsorted(ts, times, side="left")                 # strictly earlier edges
            lo = np.searchsorted(ts, times - active_window, side="left")
            cnt = hi - lo
            r = lo[:, None] + (rng.random((len(srcs), k)) * np.maximum(cnt, 1)[:, None]).astype(np.int64)
            r = np.minimum(r, len(ts) - 1)
            pick_src = rng.random((len(srcs), k)) < 0.5
            out = np.where(pick_src, src[r], tgt[r]).astype(np.int64)
            return np.where((cnt == 0)[:, None], rng.integers(0, n, size=out.shape), out)

    def bad(srcs, cands, tix):
        keys = (srcs[:, None] * n + cands) * K + tix[:, None]
        bm = np.isin(keys, forbidden) | (cands == srcs[:, None])
        if train_pairs is not None:
            bm |= np.isin(srcs[:, None] * n + cands, train_pairs)
        return bm

    c = draw(s, tt, n_neg).astype(np.int64)
    initial_rate = n_fallback = 0
    solved = False
    for it in range(max_redraws + 20):
        bm = bad(s, c, tidx)
        if it == 0:
            initial_rate = float(bm.mean())
        if not bm.any():
            solved = True
            break
        rows, cols = np.nonzero(bm)
        if it < max_redraws:
            c[rows, cols] = draw(s[rows], tt[rows], 1)[:, 0]
        else:
            if it == max_redraws:
                n_fallback = int(bm.sum())
            c[rows, cols] = rng.integers(0, n, size=len(rows))
    if not solved:
        raise ValueError(f"[{kind}] could not draw collision-free negatives; universe too small?")

    meta = {"initial_collision_rate": initial_rate,
            "fallback_to_uniform": n_fallback,
            "fallback_fraction": n_fallback / max(c.size, 1),
            "n_test_events": int(len(test)), "n_negatives": int(n_neg),
            "active_window_s": active_window if kind == "active" else None}
    if meta["fallback_fraction"] > 0.01:
        log.warning("[%s] %.1f%% of negatives fell back to uniform; this sampler is "
                    "effectively partly random here", kind, 100 * meta["fallback_fraction"])
    return NegativeSet(kind, c, seed, meta)


# ------------------------------------------------------------------ scoring
def score_events_strict(result, bundle: GraphBundle, split: TemporalSplit,
                        neg_ids: np.ndarray, max_batch: int = 4):
    """Score the test events of a trained `PolymapTGN` on FIXED negatives.

    Same maths as `trainer.score_events`, except that batches are whole
    timestamp groups (see module docstring). Returns (pos (n,), neg (n, K)).
    """
    import torch

    from ..models.pair_history import PairHistory
    from ..training.trainer import normalise_edge_attr

    model, memory = result.model, result.memory
    model.eval()
    device = torch.device(result.device)
    edge_attr, _ = normalise_edge_attr(bundle.edge_attr, split.train_idx)
    ei = torch.tensor(bundle.edge_index, dtype=torch.long, device=device)
    ea = torch.tensor(edge_attr, dtype=torch.float32, device=device)
    src_np, tgt_np = bundle.edge_index
    ts_np = bundle.timestamps
    test = split.test_idx
    K = neg_ids.shape[1]
    if neg_ids.shape[0] != len(test):
        raise ValueError("neg_ids rows must equal the number of test events")

    pos_out = np.empty(len(test), dtype=np.float64)
    neg_out = np.empty((len(test), K), dtype=np.float64)
    mem = memory.clone()
    ph = PairHistory(bundle.num_nodes)
    ph.replay(bundle.edge_index, ts_np, split.train_idx)

    with torch.no_grad():
        for sl in time_grouped_slices(ts_np[test], max_batch):
            idx_np = test[sl]
            idx = torch.as_tensor(idx_np, device=device)
            s, t = ei[0, idx], ei[1, idx]
            neg_np = neg_ids[sl]
            neg_t = torch.as_tensor(neg_np, device=device)
            times = ts_np[idx_np]
            pp = pn = None
            if model.pair_features:
                pp = torch.as_tensor(ph.features(src_np[idx_np], tgt_np[idx_np], times),
                                     dtype=torch.float32, device=device)
                pn = torch.as_tensor(
                    ph.features(np.repeat(src_np[idx_np], K), neg_np.reshape(-1),
                                np.repeat(times, K)),
                    dtype=torch.float32, device=device)
            p = model.score(mem, s, t, pp)
            ns = model.score(mem, s[:, None].expand(-1, K).reshape(-1),
                             neg_t.reshape(-1), pn).reshape(len(idx_np), K)
            pos_out[sl] = p.cpu().numpy()
            neg_out[sl] = ns.cpu().numpy()
            mem = model.advance(mem, s, t, ea[idx])
            ph.update(src_np[idx_np], tgt_np[idx_np], times)
    return pos_out, neg_out


def score_baselines_strict(bundle: GraphBundle, split: TemporalSplit, neg_ids: np.ndarray,
                           max_batch: int, names: tuple[str, ...]) -> dict[str, dict]:
    """Baselines on the same fixed negatives and the same timestamp-grouped batches."""
    from ..models import baselines_extra  # noqa: F401  (registers EdgeBank etc.)
    from ..models.baselines import ALL_BASELINES

    bad = [n for n in names if n not in ALL_BASELINES]
    if bad:
        raise ValueError(f"unknown baseline(s) {bad}; choose from {list(ALL_BASELINES)}")
    src, tgt = bundle.edge_index
    ts = bundle.timestamps
    test = split.test_idx
    k = neg_ids.shape[1]
    slices = time_grouped_slices(ts[test], max_batch)
    out: dict[str, dict] = {}
    for name in names:
        model = ALL_BASELINES[name]().fit(bundle.edge_index, ts, split.train_idx)
        pos = np.empty(len(test))
        neg = np.empty((len(test), k))
        for sl in slices:
            idx = test[sl]
            s, t, tm = src[idx], tgt[idx], ts[idx]
            pos[sl] = model.score_candidates(s, t[:, None], tm)[:, 0]
            neg[sl] = model.score_candidates(s, neg_ids[sl], tm)
            model.update(s, t, tm)
        out[name] = {"positive": pos, "negative": neg}
    return out


# ------------------------------------------------------------------ metrics
def summarize_scores(pos: np.ndarray, neg: np.ndarray, gap: np.ndarray, window_s: float,
                     mask: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    """Metric row + per-event reciprocal ranks (masked)."""
    from .metrics import classification_metrics, per_event_ranking

    if mask is not None:
        pos, neg, gap = pos[mask], neg[mask], gap[mask]
    ev = per_event_ranking(pos, neg)
    rr = ev["rr"]
    cm = classification_metrics(pos, neg.ravel())
    row = {"n_events": int(len(pos)), "mrr": float(rr.mean()),
           "hits@1": float(ev["hits@1"].mean()), "hits@10": float(ev["hits@10"].mean()),
           "roc_auc": cm["roc_auc"], "average_precision": cm["average_precision"]}
    strata = {"never_seen": np.isnan(gap), "recent": gap <= window_s, "stale": gap > window_s}
    for name, m in strata.items():
        row[f"n_{name}"] = int(m.sum())
        row[f"mrr_{name}"] = float(rr[m].mean()) if m.any() else float("nan")
    return row, rr


def block_bootstrap(d: np.ndarray, n_boot: int = 2000, n_blocks: int = 50,
                    seed: int = 0, alpha: float = 0.05) -> dict:
    """Mean of per-event differences with a contiguous-time-block bootstrap CI.

    Test events in the same burst share wallets, so single-event resampling
    understates uncertainty (same reasoning as `paired_mrr_gap`).
    """
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    nb = max(1, min(n_blocks, n))
    blocks = np.array_split(np.arange(n), nb)
    sums = np.array([d[b].sum() for b in blocks])
    sizes = np.array([len(b) for b in blocks], dtype=float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, nb, size=(n_boot, nb))
    boot = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(d.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "n_events": int(n), "n_blocks": int(nb)}
