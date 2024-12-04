"""Tests for the src/domain/shared/evaluation.py module.

Author: Mikel Sagardia
Date: 2024-12-04
"""
import pytest
from sklearn.metrics import roc_auc_score

from src.domain.shared.evaluation import (Evaluator,
                                          Metric,
                                          get_optimization_direction,
                                          OptimizationDirection)


@pytest.mark.parametrize(
    "metric, y_true, y_pred, expected",
    [
        (Metric.ACCURACY, [0, 1, 1, 0], [0, 1, 0, 0], 0.75),
        (Metric.F1, [0, 1, 1, 0], [0, 1, 0, 0], 0.7333333333333334),
        (Metric.PRECISION, [0, 1, 1, 0], [0, 1, 0, 0], 0.8333333333333333),
        (Metric.RECALL, [0, 1, 1, 0], [0, 1, 0, 0], 0.75),
        (Metric.MSE, [0, 1, 1, 0], [0, 1, 0, 0], 0.25),
    ]
)
def test_evaluate_basic_metrics(metric, y_true, y_pred, expected):
    """Test the evaluation of basic metrics."""
    evaluator = Evaluator()
    result = evaluator.evaluate(y_true, y_pred, metric)
    assert result == pytest.approx(expected, 0.001), f"Expected {expected} for metric {metric}, but got {result}"


def test_evaluate_roc_auc_binary():
    """Test the evaluation of the ROC AUC metric for binary classification."""
    y_true = [0, 1, 1, 0]
    y_pred = [0.1, 0.9, 0.4, 0.2] # Probabilities for positive class
    evaluator = Evaluator(Metric.ROC_AUC)
    result = evaluator.evaluate(y_true, y_pred)
    expected = roc_auc_score(y_true, y_pred)
    assert result == pytest.approx(expected, 0.001), f"Expected ROC AUC of {expected}, but got {result}"


def test_evaluate_roc_auc_multiclass():
    """Test the evaluation of the ROC AUC metric for multiclass classification."""
    y_true = [0, 1, 2, 1, 0, 2]
    y_pred = [
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.2, 0.6],
        [0.1, 0.6, 0.3],
        [0.9, 0.05, 0.05],
        [0.1, 0.3, 0.6],
    ] # Probabilities for each class
    evaluator = Evaluator(Metric.ROC_AUC)
    result = evaluator.evaluate(y_true, y_pred)
    expected = roc_auc_score(y_true, y_pred, multi_class="ovr")
    assert result == pytest.approx(expected, 0.001), f"Expected ROC AUC of {expected}, but got {result}"


def test_set_metric_invalid():
    """Test setting an invalid metric."""
    evaluator = Evaluator()
    with pytest.raises(ValueError):
        evaluator.set_metric("invalid_metric")


def test_get_optimization_direction():
    """Test the get_optimization_direction function."""
    # MAXIMIZE
    assert get_optimization_direction(Metric.ACCURACY) == OptimizationDirection.MAXIMIZE
    assert get_optimization_direction(Metric.F1) == OptimizationDirection.MAXIMIZE
    assert get_optimization_direction(Metric.PRECISION) == OptimizationDirection.MAXIMIZE
    assert get_optimization_direction(Metric.RECALL) == OptimizationDirection.MAXIMIZE
    assert get_optimization_direction(Metric.ROC_AUC) == OptimizationDirection.MAXIMIZE
    # MINIMIZE
    assert get_optimization_direction(Metric.MSE) == OptimizationDirection.MINIMIZE


def test_evaluator_metric_property():
    """Test the metric property of the Evaluator."""
    evaluator = Evaluator(Metric.PRECISION)
    assert evaluator.metric == Metric.PRECISION
    evaluator.set_metric(Metric.RECALL)
    assert evaluator.metric == Metric.RECALL


def test_evaluator_optimization_direction_property():
    """Test the optimization_direction property of the Evaluator."""
    evaluator = Evaluator(Metric.MSE)
    assert evaluator.optimization_direction == OptimizationDirection.MINIMIZE
    evaluator.set_metric(Metric.ACCURACY)
    assert evaluator.optimization_direction == OptimizationDirection.MAXIMIZE
