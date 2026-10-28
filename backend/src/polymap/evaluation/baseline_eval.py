"""Score baselines on exactly the negatives the TGNN was scored on.

The negatives are drawn once, inside `trainer.score_events`, and returned as
`neg_ids`. Re-drawing them here (even with the same seed) would let the
comparison drift if either side's draw order ever changed, so we reuse them.

Needs only numpy: no torch.
"""
from __future__ import annotations

import numpy as np

from ..models.baselines import ALL_BASELINES


def score_baselines(bundle, split, neg_ids: np.ndarray, batch_size: int,
                    names: tuple[str, ...] | None = None) -> dict[str, dict]:
    """Return {name: {"positive": (n,), "negative": (n, k)}} over the test events.

    Test events are processed in the same chronological batches as the TGNN:
    score the whole batch first, then absorb it. That keeps every baseline from
    seeing an event before it is scored.
    """
    names = tuple(names) if names else tuple(ALL_BASELINES)
    unknown = [n for n in names if n not in ALL_BASELINES]
    if unknown:
        raise ValueError(f"unknown baseline(s) {unknown}; choose from {list(ALL_BASELINES)}")

    src, tgt = bundle.edge_index
    ts = bundle.timestamps
    test = split.test_idx
    if neg_ids.shape[0] != len(test):
        raise ValueError(f"neg_ids has {neg_ids.shape[0]} rows but there are {len(test)} test events")
    k = neg_ids.shape[1]

    out: dict[str, dict] = {}
    for name in names:
        model = ALL_BASELINES[name]().fit(bundle.edge_index, ts, split.train_idx)
        pos = np.empty(len(test), dtype=np.float64)
        neg = np.empty((len(test), k), dtype=np.float64)
        for start in range(0, len(test), batch_size):
            sl = slice(start, start + batch_size)
            idx = test[sl]
            s, t, tm = src[idx], tgt[idx], ts[idx]
            pos[sl] = model.score_candidates(s, t[:, None], tm)[:, 0]
            neg[sl] = model.score_candidates(s, neg_ids[sl], tm)
            model.update(s, t, tm)
        out[name] = {"positive": pos, "negative": neg}
    return out


def evaluate_scores(pos: np.ndarray, neg: np.ndarray, gap_s: np.ndarray, window_s: float) -> dict:
    """The full metric block for one scorer: overall + stratified by pair recency.

    Used for the TGNN and every baseline alike so the numbers are comparable.
    """
    from .metrics import classification_metrics, ranking_metrics

    strata = {"never_seen_pair": np.isnan(gap_s),
              "recent_pair": gap_s <= window_s,       # NaN compares False
              "stale_pair": gap_s > window_s}
    by_recency = {name: {"n": int(m.sum()),
                         **classification_metrics(pos[m], neg[m].ravel()),
                         **ranking_metrics(pos[m], neg[m])}
                  for name, m in strata.items() if m.sum() > 0}
    return {**classification_metrics(pos, neg.ravel()), **ranking_metrics(pos, neg),
            "by_recency": by_recency}


def paired_mrr_gap(pos_a, neg_a, pos_b, neg_b, n_boot: int = 1000, n_blocks: int = 50,
                   seed: int = 0) -> dict:
    """MRR(a) - MRR(b) on the same test events, with a block-bootstrap 95% interval.

    Events are resampled in contiguous time blocks, not one by one, because
    neighbouring test events share wallets and bursts. Resampling single events
    would understate the uncertainty. Both models are scored on the same rows,
    so the difference is paired.
    """
    from .metrics import per_event_ranking

    ra = per_event_ranking(pos_a, neg_a)["rr"]
    rb = per_event_ranking(pos_b, neg_b)["rr"]
    d = ra - rb
    n = len(d)
    nb = max(1, min(n_blocks, n))
    blocks = np.array_split(np.arange(n), nb)
    sums = np.array([d[b].sum() for b in blocks])
    sizes = np.array([len(b) for b in blocks], dtype=float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, nb, size=(n_boot, nb))
    boot = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mrr_gap": float(d.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "n_events": int(n), "n_blocks": int(nb)}
