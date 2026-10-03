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


def ranking_metrics(pos_scores: np.ndarray, neg_scores: np.ndarray,
                    ks=(1, 5, 10)) -> dict:
    """MRR and Hits@k where each row is one positive against N negatives."""
    pos_scores = np.asarray(pos_scores).reshape(-1, 1)
    neg_scores = np.asarray(neg_scores)
    if neg_scores.ndim == 1:
        neg_scores = neg_scores.reshape(-1, 1)
    # rank = 1 + how many negatives beat the positive
    ranks = 1 + (neg_scores > pos_scores).sum(axis=1)
    out = {"mrr": float(np.mean(1.0 / ranks)), "mean_rank": float(np.mean(ranks))}
    for k in ks:
        out[f"hits@{k}"] = float(np.mean(ranks <= k))
    return out
