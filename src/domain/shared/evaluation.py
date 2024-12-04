"""Evaluation module.

This module contains the Evaluator class,
which is used to evaluate model predictions using various metrics.
The metrics are defined in the Metric enum.
The advantage of using this dedicated module is that it centralizes and standardizes
any model/estimator metrics and evaluation.

Author: Mikel Sagardia
Date: 2024-12-04
"""
from typing import Any, Optional
from enum import Enum
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             precision_score,
                             recall_score,
                             roc_auc_score,
                             mean_squared_error)


class Metric(Enum):  # noqa: D101
    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    ROC_AUC = "roc_auc"
    MSE = "mse"
    # Define any other metrics you need here
    # For instance, specific image segmentation or object detection metrics
    # like IOU, etc., which can be implemented here as functions
    # and used by Evaluator below.


class OptimizationDirection(Enum):  # noqa: D101
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


def get_optimization_direction(metric: Metric) -> OptimizationDirection:
    """Get the optimization direction for a given metric."""
    optimization_direction = OptimizationDirection.MAXIMIZE
    # Extend this for any metric which needs minimization
    if metric == Metric.MSE:
        optimization_direction = OptimizationDirection.MINIMIZE

    return optimization_direction


def get_metric_from_string(metric_string: str) -> Metric:
    """Get the Metric enum value from a string."""
    if metric_string is None:
        return Metric.F1
    else:
        try:
            return Metric(metric_string.lower())
        except ValueError:
            raise ValueError(f"Unsupported metric: {metric_string}") from None


class Evaluator:
    """A class to evaluate model predictions using various metrics.

    Usage example:

        # The type/format of y and y_pred depends on the task
        # and the implementation inside Evaluator should account for it
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]

        evaluator = Evaluator(Metric.ACCURACY)
        result = evaluator.evaluate(y_true, y_pred)
        print(result) # 0.75

        result = evaluator.evaluate(y_true, y_pred, Metric.F1)
        print(result) # 0.6666666666666666
    """

    def __init__(self, metric: Metric = Metric.F1):
        self.set_metric(metric)

    def set_metric(self, metric: Metric) -> None:  ## noqa: D102
        if isinstance(metric, Metric):
            self._metric = metric
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        self._optimization_direction = get_optimization_direction(self._metric)

    def evaluate(self, y: Any, y_pred: Any, metric: Optional[Metric] = None) -> Any:
        """Evaluate the predictions using the specified metric; main entry-point."""
        if metric is not None:
            self.set_metric(metric)
        # else: the Evaluator _metric is used

        # Implement the evaluation logic here
        # ...
        if self._metric == Metric.ACCURACY:
            return accuracy_score(y, y_pred)
        elif self._metric == Metric.F1:
            return f1_score(y, y_pred, average="weighted")
        elif self._metric == Metric.PRECISION:
            return precision_score(y, y_pred, average="weighted")
        elif self._metric == Metric.RECALL:
            return recall_score(y, y_pred, average="weighted")
        elif self._metric == Metric.ROC_AUC:
            # Watch out: if binary classification, y_pred should be the probability of the positive class, i.e.,
            # y_pred = model.predict(X)  # noqa: ERA001
            # If multi-class classification, y_pred should be the probabilities of each class, i.e.
            # y_pred = model.predict_proba(X)  # noqa: ERA001
            if len(set(y)) == 2: # Binary classification  # noqa: PLR2004
                return roc_auc_score(y, y_pred)
            else: # Multi-class classification
                return roc_auc_score(y, y_pred, multi_class="ovr")
        elif self._metric == Metric.MSE:
            return mean_squared_error(y, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {self._metric}")

    @property
    def metric(self):  # noqa: D102
        return self._metric

    @property
    def optimization_direction(self):  # noqa: D102
        return self._optimization_direction
