"""Metrics. Ranking metrics lead, because they compare orderings rather than
score scales that differ between methods."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def classification_metrics(pos: np.ndarray, neg: np.ndarray) -> dict:
    y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    s = np.concatenate([pos, neg])
    out = {"roc_auc": float(roc_auc_score(y, s)), "average_precision": float(average_precision_score(y, s))}
    if out["roc_auc"] < 0.5:
        # Worth flagging loudly: under the hard-negative protocol a history-based
        # baseline is *expected* to land here, and the paper must explain why
        # rather than present it as a clean win.
        out["below_chance"] = True
        out["note"] = ("AUC below 0.5 means the ordering is inverted. Under hard "
                       "negatives this is expected for history-based scorers, "
                       "since negatives are chosen using that same signal.")
    return out


def per_event_ranking(pos_scores: np.ndarray, neg_scores: np.ndarray,
                      ks=(1, 5, 10)) -> dict[str, np.ndarray]:
    """Per-event reciprocal rank and hits@k, with ties broken at random (in expectation).

    If g negatives score strictly above the positive and e tie with it, the
    positive's rank is uniform on {1+g, ..., 1+g+e}. Counting ties as wins for
    the positive would hand count-based baselines (which score most negatives
    exactly 0) a large, spurious boost; counting them as losses would punish
    them. The expectation is the neutral choice, and for continuous scores
    (e == 0) it reduces to the usual rank.
    """
    pos = np.asarray(pos_scores, dtype=np.float64).reshape(-1, 1)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if neg.ndim == 1:
        neg = neg.reshape(-1, 1)
    g = (neg > pos).sum(axis=1)
    e = (neg == pos).sum(axis=1)
    # E[1/r] for r uniform on 1+g..1+g+e  =  (H(1+g+e) - H(g)) / (e+1)
    H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, neg.shape[1] + 2))])
    out = {"rr": (H[g + e + 1] - H[g]) / (e + 1),
           "mean_rank": g + 1 + e / 2.0}
    for k in ks:
        # P(rank <= k) = clip((k - g) / (e + 1), 0, 1)
        out[f"hits@{k}"] = np.clip((k - g) / (e + 1.0), 0.0, 1.0)
    return out


def ranking_metrics(pos_scores: np.ndarray, neg_scores: np.ndarray,
                    ks=(1, 5, 10)) -> dict:
    """MRR and Hits@k where each row is one positive against N negatives.

    Ties are resolved in expectation (see `per_event_ranking`).
    """
    ev = per_event_ranking(pos_scores, neg_scores, ks)
    out = {"mrr": float(ev["rr"].mean()), "mean_rank": float(ev["mean_rank"].mean())}
    for k in ks:
        out[f"hits@{k}"] = float(ev[f"hits@{k}"].mean())
    return out
