"""
Tests for CVA metrics module.
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cva.metrics import (
    compute_accuracy,
    compute_group_accuracy,
    compute_worst_group_accuracy,
    compute_confusion_matrix,
    compute_balanced_accuracy,
    compute_flip_rate,
    compute_probability_change,
    compute_confidence_change,
    compute_jsd,
    compute_all_classification_metrics,
    compute_all_intervention_metrics,
)


class TestClassificationMetrics:
    """Tests for classification metrics."""

    def test_accuracy_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert compute_accuracy(y_true, y_pred) == 1.0

    def test_accuracy_half(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert compute_accuracy(y_true, y_pred) == 0.5

    def test_group_accuracy(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])
        groups = np.array([0, 0, 1, 1, 2, 2])

        group_accs = compute_group_accuracy(y_true, y_pred, groups)
        assert group_accs[0] == 1.0   # 2/2 correct
        assert group_accs[1] == 0.5   # 1/2 correct
        assert group_accs[2] == 0.5   # 1/2 correct

    def test_worst_group_accuracy(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1])
        groups = np.array([0, 0, 1, 1])

        worst_acc, worst_group = compute_worst_group_accuracy(y_true, y_pred, groups)
        assert worst_acc == 0.5
        assert worst_group == 1

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=2)
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 1  # True negative
        assert cm[0, 1] == 1  # False positive
        assert cm[1, 0] == 1  # False negative
        assert cm[1, 1] == 1  # True positive

    def test_balanced_accuracy(self):
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 1])
        bal_acc = compute_balanced_accuracy(y_true, y_pred)
        # Class 0: 4/4 = 1.0, Class 1: 1/2 = 0.5, balanced = 0.75
        assert abs(bal_acc - 0.75) < 1e-6

    def test_all_classification_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1])
        groups = np.array([0, 1, 2, 3])

        metrics = compute_all_classification_metrics(y_true, y_pred, groups)
        assert "accuracy" in metrics
        assert "group_accuracy" in metrics
        assert "worst_group_accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert metrics["accuracy"] == 0.75


class TestInterventionMetrics:
    """Tests for intervention metrics."""

    def test_flip_rate(self):
        preds_orig = np.array([0, 0, 1, 1])
        preds_mod = np.array([0, 1, 1, 0])
        assert compute_flip_rate(preds_orig, preds_mod) == 0.5

    def test_flip_rate_all_same(self):
        preds = np.array([0, 0, 1, 1])
        assert compute_flip_rate(preds, preds) == 0.0

    def test_probability_change(self):
        probs_orig = np.array([[0.9, 0.1], [0.3, 0.7]])
        probs_mod = np.array([[0.5, 0.5], [0.4, 0.6]])
        y_true = np.array([0, 1])

        changes = compute_probability_change(probs_orig, probs_mod, y_true)
        # Sample 0: 0.9 - 0.5 = 0.4 (true class prob decreased)
        # Sample 1: 0.7 - 0.6 = 0.1
        assert abs(changes[0] - 0.4) < 1e-6
        assert abs(changes[1] - 0.1) < 1e-6

    def test_confidence_change(self):
        probs_orig = np.array([[0.8, 0.2], [0.3, 0.7]])
        probs_mod = np.array([[0.5, 0.5], [0.4, 0.6]])

        changes = compute_confidence_change(probs_orig, probs_mod)
        # Sample 0: pred=0, conf 0.8 -> 0.5, change = 0.3
        # Sample 1: pred=1, conf 0.7 -> 0.6, change = 0.1
        assert abs(changes[0] - 0.3) < 1e-6
        assert abs(changes[1] - 0.1) < 1e-6

    def test_jsd_identical(self):
        probs = np.array([[0.5, 0.5], [0.8, 0.2]])
        jsd = compute_jsd(probs, probs)
        # JSD of identical distributions should be ~0
        assert np.all(jsd < 1e-6)

    def test_jsd_different(self):
        probs_orig = np.array([[1.0, 0.0]])
        probs_mod = np.array([[0.0, 1.0]])
        jsd = compute_jsd(probs_orig, probs_mod)
        # JSD of maximally different distributions should be ~ln(2) ≈ 0.693
        assert jsd[0] > 0.5

    def test_all_intervention_metrics(self):
        probs_orig = np.array([[0.9, 0.1], [0.3, 0.7], [0.8, 0.2], [0.2, 0.8]])
        probs_mod = np.array([[0.4, 0.6], [0.5, 0.5], [0.7, 0.3], [0.3, 0.7]])
        y_true = np.array([0, 1, 0, 1])
        groups = np.array([0, 1, 0, 1])

        metrics = compute_all_intervention_metrics(probs_orig, probs_mod, y_true, groups)
        assert "flip_rate" in metrics
        assert "mean_probability_change" in metrics
        assert "mean_jsd" in metrics
        assert "per_group" in metrics
        assert 0 in metrics["per_group"]
        assert 1 in metrics["per_group"]
