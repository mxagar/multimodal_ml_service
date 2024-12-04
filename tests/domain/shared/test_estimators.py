"""Tests for the src/domain/shared/estimators.py module.

Author: Mikel Sagardia
Date: 2024-12-04
"""
import pytest
import numpy as np

from src.domain.shared.estimators import (RuleBasedEstimator,
                                          SklearnPipeEstimator)


@pytest.mark.parametrize(
    "threshold, X, expected",
    [
        (0.5, [0.3, 0.7, 0.5], [0, 1, 0]),
        (0.6, [0.3, 0.7, 0.5], [0, 1, 0]),
        (0.2, [0.3, 0.7, 0.5], [1, 1, 1])
    ]
)
def test_dummy_rule_based_model_predict(dummy_rule_based_model, threshold, X, expected):
    """Test the predict method of the DummyRuleBasedModel."""
    dummy_rule_based_model.set_params(threshold=threshold)
    predictions = dummy_rule_based_model.predict(X)
    assert np.array_equal(predictions, expected),\
        f"Expected predictions to be {expected}, but got {predictions}."


@pytest.mark.parametrize("params", [
    {"classifier": {"C": 0.1}},
    {"classifier": {"C": 1.0}},
])
def test_sklearn_pipeline_estimator_fit_predict(dummy_sklearn_pipeline, params):
    """Test the fit and predict methods of the SklearnPipeEstimator."""
    X_train = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]])
    y_train = np.array([0, 1, 1])
    X_test = np.array([[0.3, 0.4], [0.6, 0.7]])

    estimator = SklearnPipeEstimator(dummy_sklearn_pipeline)
    estimator.set_params(nested=True, **params)
    estimator.fit(X_train, y_train)

    predictions = estimator.predict(X_test)
    assert len(predictions) == len(X_test),\
        "Predictions length should match the test data length."


def test_sklearn_pipeline_get_set_params(dummy_sklearn_pipeline):
    """Test the get_params and set_params methods of the SklearnPipeEstimator."""
    estimator = SklearnPipeEstimator(dummy_sklearn_pipeline)
    params = {"classifier": {"C": 0.5}}
    estimator.set_params(nested=True, **params)
    retrieved_params = estimator.get_params(nested=True)
    assert retrieved_params["classifier"]["C"] == 0.5,\
        "The retrieved parameters should match the set values."


def test_rule_based_estimator_fit_predict(dummy_rule_based_model):
    """Test the fit and predict methods of the RuleBasedEstimator."""
    X_test = [0.3, 0.7, 0.5]
    estimator = RuleBasedEstimator(dummy_rule_based_model)
    estimator.set_params(threshold=0.5)
    predictions = estimator.predict(X_test)
    #assert predictions == np.array([0, 1, 0]),\
    assert np.array_equal(predictions, np.array([0, 1, 0])),\
        "The predictions should match the expected rule-based output."


def test_rule_based_estimator_get_set_params(dummy_rule_based_model):
    """Test the get_params and set_params methods of the RuleBasedEstimator."""
    estimator = RuleBasedEstimator(dummy_rule_based_model)
    params = {"threshold": 0.8}
    estimator.set_params(**params)
    retrieved_params = estimator.get_params()
    assert retrieved_params["threshold"] == 0.8,\
        "The retrieved parameters should match the set values."
