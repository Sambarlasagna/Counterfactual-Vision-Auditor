"""
Metrics for CVA experiments.

Provides:
  - Classification metrics (accuracy, group accuracy, worst-group, confusion matrix)
  - Intervention metrics (flip rate, confidence change, probability delta, JSD)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
)


logger = logging.getLogger("cva")


# ── Classification Metrics ─────────────────────────────────────────────

def compute_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute overall accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_group_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Dict[int, float]:
    """
    Compute per-group accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        groups: Group assignments for each sample.

    Returns:
        Dict mapping group_id -> accuracy.
    """
    group_accs = {}
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() > 0:
            group_accs[int(g)] = float(accuracy_score(y_true[mask], y_pred[mask]))
        else:
            group_accs[int(g)] = float("nan")
    return group_accs


def compute_worst_group_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Tuple[float, int]:
    """
    Compute worst-group accuracy.

    Returns:
        Tuple of (worst_accuracy, worst_group_id).
    """
    group_accs = compute_group_accuracy(y_true, y_pred, groups)
    # Filter out NaN values
    valid = {g: a for g, a in group_accs.items() if not np.isnan(a)}
    if not valid:
        return float("nan"), -1
    worst_group = min(valid, key=valid.get)
    return valid[worst_group], worst_group


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 2,
) -> np.ndarray:
    """Compute confusion matrix."""
    return sk_confusion_matrix(
        y_true, y_pred,
        labels=list(range(num_classes)),
    )


def compute_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute balanced accuracy (macro-averaged recall)."""
    return float(balanced_accuracy_score(y_true, y_pred))


def compute_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute macro F1 score."""
    return float(f1_score(y_true, y_pred, average="macro"))


def compute_all_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    num_classes: int = 2,
) -> Dict[str, Any]:
    """
    Compute all classification metrics at once.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        groups: Group assignments.
        num_classes: Number of classes.

    Returns:
        Dictionary with all metric values.
    """
    group_accs = compute_group_accuracy(y_true, y_pred, groups)
    worst_acc, worst_group = compute_worst_group_accuracy(y_true, y_pred, groups)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)

    # Per-group sample counts
    group_counts = {int(g): int((groups == g).sum()) for g in np.unique(groups)}

    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "balanced_accuracy": compute_balanced_accuracy(y_true, y_pred),
        "macro_f1": compute_macro_f1(y_true, y_pred),
        "group_accuracy": group_accs,
        "worst_group_accuracy": worst_acc,
        "worst_group_id": worst_group,
        "confusion_matrix": cm.tolist(),
        "group_counts": group_counts,
        "total_samples": len(y_true),
    }


# ── Intervention Metrics ───────────────────────────────────────────────

def compute_probability_change(
    probs_original: np.ndarray,
    probs_intervened: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """
    Compute true-class probability change per example.

    p(y_true | x) - p(y_true | T(x))

    Positive values mean the intervention reduced confidence in the true class.

    Args:
        probs_original: (N, C) probability distributions for original images.
        probs_intervened: (N, C) probability distributions for intervened images.
        y_true: (N,) true class labels.

    Returns:
        (N,) array of probability changes.
    """
    N = len(y_true)
    original_true_probs = probs_original[np.arange(N), y_true]
    intervened_true_probs = probs_intervened[np.arange(N), y_true]
    return original_true_probs - intervened_true_probs


def compute_flip_rate(
    preds_original: np.ndarray,
    preds_intervened: np.ndarray,
) -> float:
    """
    Compute the prediction flip rate.

    Fraction of examples whose predicted class changes after intervention.

    Args:
        preds_original: (N,) original predicted classes.
        preds_intervened: (N,) intervened predicted classes.

    Returns:
        Float between 0 and 1.
    """
    return float(np.mean(preds_original != preds_intervened))


def compute_confidence_change(
    probs_original: np.ndarray,
    probs_intervened: np.ndarray,
) -> np.ndarray:
    """
    Compute confidence change per example.

    Difference in the probability of the originally predicted class.

    Args:
        probs_original: (N, C) probability distributions for original images.
        probs_intervened: (N, C) probability distributions for intervened images.

    Returns:
        (N,) array of confidence changes (positive = confidence decreased).
    """
    preds_original = probs_original.argmax(axis=1)
    N = len(preds_original)
    original_conf = probs_original[np.arange(N), preds_original]
    intervened_conf = probs_intervened[np.arange(N), preds_original]
    return original_conf - intervened_conf


def compute_jsd(
    probs_original: np.ndarray,
    probs_intervened: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute Jensen-Shannon Divergence per example.

    Args:
        probs_original: (N, C) probability distributions.
        probs_intervened: (N, C) probability distributions.
        eps: Small constant for numerical stability.

    Returns:
        (N,) array of JSD values.
    """
    p = probs_original + eps
    q = probs_intervened + eps

    # Normalize
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m), axis=1)
    kl_qm = np.sum(q * np.log(q / m), axis=1)

    return 0.5 * (kl_pm + kl_qm)


def compute_all_intervention_metrics(
    probs_original: np.ndarray,
    probs_intervened: np.ndarray,
    y_true: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute all intervention metrics.

    Args:
        probs_original: (N, C) original prediction probabilities.
        probs_intervened: (N, C) intervened prediction probabilities.
        y_true: (N,) true labels.
        groups: (N,) optional group assignments.

    Returns:
        Dictionary with all intervention metric values.
    """
    preds_original = probs_original.argmax(axis=1)
    preds_intervened = probs_intervened.argmax(axis=1)

    prob_changes = compute_probability_change(probs_original, probs_intervened, y_true)
    conf_changes = compute_confidence_change(probs_original, probs_intervened)
    jsd_values = compute_jsd(probs_original, probs_intervened)

    results = {
        "flip_rate": compute_flip_rate(preds_original, preds_intervened),
        "mean_probability_change": float(np.mean(prob_changes)),
        "std_probability_change": float(np.std(prob_changes)),
        "mean_confidence_change": float(np.mean(conf_changes)),
        "std_confidence_change": float(np.std(conf_changes)),
        "mean_jsd": float(np.mean(jsd_values)),
        "median_jsd": float(np.median(jsd_values)),
    }

    # Per-group intervention metrics
    if groups is not None:
        per_group = {}
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() == 0:
                continue
            per_group[int(g)] = {
                "flip_rate": compute_flip_rate(
                    preds_original[mask], preds_intervened[mask]
                ),
                "mean_probability_change": float(np.mean(prob_changes[mask])),
                "mean_confidence_change": float(np.mean(conf_changes[mask])),
                "mean_jsd": float(np.mean(jsd_values[mask])),
                "count": int(mask.sum()),
            }
        results["per_group"] = per_group

    return results
