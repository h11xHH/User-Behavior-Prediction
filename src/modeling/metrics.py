"""modeling/metrics.py

Evaluation metrics:
1. pr_auc   area under precision-recall
2. roc_auc  area under ROC
3. P@k, R@k precision and recall among the top-k highest-scored candidates
           (the operational "recommend k items" view)

Shared by the baseline and the later models.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_KS = (100, 500, 1000)


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    """Precision and recall among the top-k highest scores (ties broken by order)."""
    k = min(k, len(scores))
    top = np.argsort(scores)[::-1][:k]
    hits = float(y_true[top].sum())
    total_pos = float(y_true.sum())
    precision = hits / k if k else 0.0
    recall = hits / total_pos if total_pos else 0.0
    return precision, recall


def evaluate(y_true, scores, ks: tuple = DEFAULT_KS) -> dict:
    """Compute the metric bundle for one (labels, scores) pair.

    Returns a dict with n, positives, pr_auc, roc_auc, and P@k / R@k for each k.
    ROC/PR-AUC are guarded to NaN if only one class is present.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    result = {"n": int(len(y)), "positives": int(y.sum())}
    if y.sum() == 0 or y.sum() == len(y):
        result["pr_auc"] = float("nan")
        result["roc_auc"] = float("nan")
    else:
        result["pr_auc"] = float(average_precision_score(y, s))
        result["roc_auc"] = float(roc_auc_score(y, s))
    for k in ks:
        precision, recall = precision_recall_at_k(y, s, k)
        result[f"P@{k}"] = precision
        result[f"R@{k}"] = recall
    return result