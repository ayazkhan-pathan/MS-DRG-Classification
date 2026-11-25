"""Single-label classification metrics for DRG prediction.

These are the metrics DRG-LLaMA reports for MS-DRG prediction: top-1, top-5 and top-10
accuracy and macro-F1. Macro-F1 averages the per-DRG F1 over the DRGs present in the true
labels, as DRG-LLaMA's evaluation code does (scikit-learn's ``f1_score`` with
``labels=np.unique(y_true)``). Predicting a DRG that is absent from the labels still counts
against the true DRG's recall, but adds no zero to the average. DRG-LLaMA also reports macro-
and micro-AUC; they are not computed here. The test suite checks each function against
scikit-learn.

A true label may be a class id the scores do not cover (a DRG never seen in training); it
then counts as a miss for every metric.
"""

from __future__ import annotations

import numpy as np

TOP_KS = (5, 10)


def _as_scores_and_labels(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D (notes x classes), got shape {scores.shape}")
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(f"{scores.shape[0]} rows of scores but {labels.shape[0]} labels")
    if scores.shape[0] == 0:
        raise ValueError("no notes to score")
    return scores, labels.astype(np.int64)


def top_k_accuracy(scores, labels, k: int = 1) -> float:
    """Share of notes whose true class is among the ``k`` highest-scoring classes."""
    scores, labels = _as_scores_and_labels(scores, labels)
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    return float(np.mean(np.any(top == labels[:, None], axis=1)))


def macro_f1(predictions, labels, classes=None) -> float:
    """Unweighted mean of per-class F1.

    ``classes`` defaults to every class found in ``labels``. A class with no true and no
    predicted notes scores 0, as with scikit-learn's ``zero_division=0``.
    """
    y_pred = np.asarray(predictions).ravel()
    y_true = np.asarray(labels).ravel()
    if y_pred.shape != y_true.shape:
        raise ValueError(f"{y_pred.shape[0]} predictions but {y_true.shape[0]} labels")
    if y_true.size == 0:
        raise ValueError("no notes to score")
    classes = np.unique(y_true) if classes is None else np.unique(np.asarray(classes))
    n = classes.size

    def index_of(values):
        idx = np.searchsorted(classes, values)
        known = idx < n
        known[known] = classes[idx[known]] == values[known]
        return idx, known

    t_idx, t_known = index_of(y_true)
    p_idx, p_known = index_of(y_pred)
    hit = (y_true == y_pred) & t_known
    true_positive = np.bincount(t_idx[hit], minlength=n)
    true_count = np.bincount(t_idx[t_known], minlength=n)
    pred_count = np.bincount(p_idx[p_known], minlength=n)
    # F1 = 2 TP / (2 TP + FP + FN), and 2 TP + FP + FN = true count + predicted count.
    denominator = true_count + pred_count
    f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(n, dtype=np.float64),
        where=denominator > 0,
    )
    return float(f1.mean())


def classification_metrics(scores, labels, top_ks=TOP_KS) -> dict[str, float]:
    """Top-1 and top-k accuracy and macro-F1 from class scores (logits or probabilities)."""
    scores, labels = _as_scores_and_labels(scores, labels)
    predictions = scores.argmax(axis=1)
    metrics = {"top1_accuracy": float(np.mean(predictions == labels))}
    for k in top_ks:
        metrics[f"top{k}_accuracy"] = top_k_accuracy(scores, labels, k)
    metrics["macro_f1"] = macro_f1(predictions, labels)
    return metrics


def compute_metrics(eval_prediction) -> dict[str, float]:
    """``compute_metrics`` hook for the Hugging Face Trainer."""
    scores, labels = eval_prediction[0], eval_prediction[1]
    if isinstance(scores, (tuple, list)):
        scores = scores[0]
    return classification_metrics(scores, labels)
