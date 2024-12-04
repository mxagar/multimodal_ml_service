"""This module provides classes for training and hyperparameter optimization of estimators.

Two classes are implemented:

- TrainingArguments: A class to hold hyperparameters and configuration settings.
- Trainer: A class to handle the training and hyperparameter optimization of an estimator.

Any Estimator can be trained in the Trainer class,
which can also perform hyperparameter optimization using Optuna.
An instance of the TrainingArguments class is passed to an instance of a Trainer.
The class TrainingArguments allows for fixed hyperparameter values and hyperparameter search spaces.
Additionally, the Trainer can log metrics and models to a ModelTracker.

Usually, the Trainer class is used in the TrainingPipeline class, defined in pipelines.py.

Author: Mikel Sagardia
Date: 2024-12-04
"""
import yaml
from typing import Dict, Any, Optional, Union
import pathlib

import random
import math

import optuna

from src.adapters.logger import Logger
from src.adapters.tracker import ModelTracker
from src.domain.shared.estimators import Estimator
from src.domain.shared.evaluation import (Metric,
                                          get_optimization_direction)


class TrainingArguments:
    """TrainingArguments to hold hyperparameters and configuration settings.

    We can define both fixed hyperparameter values and hyperparameter search spaces.
    The latter is done using Optuna's suggest methods, via a dictionary with the following keys:
    - suggest: Type of parameter to suggest (int, float, categorical)
    - low: Lower bound of the search space
    - high: Upper bound of the search space
    - log: Whether to sample in log space (default: False)
    For both cases, the key names are arbitrary, but the must be accepted by the model.

    Usage:

        args = TrainingArguments(
            batch_size=32,
            learning_rate={"suggest": "float", "low": 1e-5, "high": 1e-1, "log": True},
            epochs=10,
            feature_param_1={"suggest": "int", "low": 1, "high": 10}
            optimizer={"suggest": "categorical", "choices": ["adam", "sgd", "rmsprop"]}
        )

        print(args["batch_size"]) # 32

        params = {
            'xgb': {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_estimators": {"suggest": "int", "low": 50, "high": 300}
            }
        }

        args = TrainingArguments(**params, nested=True)
        # If nested=True, the parameters are unpacked to comply with the naming convention in sklearn Pipelines
        print(args["xgb__eval_metric"]) # "logloss"

        args = TrainingArguments.from_dict({"batch_size": 64, "epochs": 20})
        args = TrainingArguments.from_yaml("path/to/args.yaml")
        args["batch_size"] # 32
        hps_fixed = args.get_fixed_hyperparameters() # {'batch_size': 32, 'epochs': 20}
        hps_tunable_sampled = args.sample_tunable_hyperparameter_space()
        # {'learning_rate': 0.0001, 'feature_param_1': 5, 'optimizer': 'adam'}

    Example args.yaml:

        batch_size: 32
        learning_rate:
            suggest: float
            low: 1e-5
            high: 1e-1
            log: true
        epochs: 10
        feature_param_1:
            suggest: int
            low: 1
            high: 10
        optimizer:
            suggest: categorical
            choices:
                - adam
                - sgd
                - rmsprop
    """

    def __init__(self,
                 metric: Metric = Metric.F1,
                 nested=False,
                 **kwargs):
        """Initialize the TrainingArguments with hyperparameters and configuration settings."""
        self._metric = metric
        self._nested = nested
        self.params = kwargs # Hyperparameters
        if self._nested:
            params = self._unpack_params(kwargs)
            self.params = params

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "TrainingArguments":  # noqa: D102
        # FIXME: Metric is missing!
        return cls(**params)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingArguments":  # noqa: D102
        # FIXME: Metric is missing!
        with open(yaml_path, "r") as file:
            params = yaml.safe_load(file)
        return cls.from_dict(params)

    def _unpack_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Unpack nested parameter dictionaries to comply with the parameter naming convention in sklearn Pipelines.

        Basically, the dictionary objects are flattened so that each
        key is prefixed with the parent key and '__' (double underscore).
        This feature is necessary for SklearnPipeEstimator.

        Input example:
            params = {
                'xgb': {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "n_estimators": {"suggest": "int", "low": 50, "high": 300},
                    "max_depth": {"suggest": "int", "low": 3, "high": 10},
                    "learning_rate": {"suggest": "float", "low": 0.01, "high": 0.3, "log": True}
                }
            }

        Output example:
            params = {
                "xgb__objective": "binary:logistic",
                "xgb__eval_metric": "logloss",
                "xgb__n_estimators": {"suggest": "int", "low": 50, "high": 300},
                "xgb__max_depth": {"suggest": "int", "low": 3, "high": 10},
                "xgb__learning_rate": {"suggest": "float", "low": 0.01, "high": 0.3, "log": True}
            }
        """
        unpacked_params = {}
        for key, value in params.items():
            if isinstance(value, dict) and "suggest" not in value:
                for sub_key, sub_value in value.items():
                    unpacked_params[f"{key}__{sub_key}"] = sub_value
            else:
                unpacked_params[key] = value
        return unpacked_params

    def get_tunable_hyperparameter_space(self, trial) -> Dict[str, Any]:
        """Get a sample of the tunable hyperparameter space for a given Optuna trial."""
        hyperparameters = {}
        for key, value in self.params.items():
            if isinstance(value, dict) and "suggest" in value:
                # If a dict has a 'suggest' key, it's treated as an Optuna parameter definition
                suggest_type = value["suggest"]
                if suggest_type == "int":
                    hyperparameters[key] = trial.suggest_int(
                        key, value["low"], value["high"], log=value.get("log", False))
                elif suggest_type == "float":
                    hyperparameters[key] = trial.suggest_float(
                        key, value["low"], value["high"], log=value.get("log", False))
                elif suggest_type == "categorical":
                    hyperparameters[key] = trial.suggest_categorical(
                        key, value["choices"])
        return hyperparameters

    def sample_tunable_hyperparameter_space(self) -> Dict[str, Any]:
        """Sample the tunable hyperparameter space to get a random set of hyperparameters."""
        hyperparameters = {}
        for key, value in self.params.items():
            if isinstance(value, dict) and "suggest" in value:
                # If a dict has a 'suggest' key, it's treated as an Optuna parameter definition
                suggest_type = value["suggest"]
                if suggest_type == "int":
                    hyperparameters[key] = random.randint(value["low"], value["high"])
                elif suggest_type == "float":
                    if value.get("log", False):
                        hyperparameters[key] = 10 ** random.uniform(math.log10(value["low"]), math.log10(value["high"]))
                    else:
                        hyperparameters[key] = random.uniform(value["low"], value["high"])
                elif suggest_type == "categorical":
                    hyperparameters[key] = random.choice(value["choices"])
        return hyperparameters

    def get_fixed_hyperparameters(self) -> Dict[str, Any]:  # noqa: D102
        return {key: value for key, value in self.params.items() if not isinstance(value, dict)}

    def update(self, params: Dict[str, Any]) -> None:  # noqa: D102
        # This will update any existing keys and append any new ones!
        self.params.update(params)

    def __getitem__(self, key: str) -> Any:
        """Get a hyperparameter value by key."""
        return self.params.get(key)

    def __str__(self) -> str:
        """Convert the TrainingArguments to a string representation."""
        return str(self.params)

    def set_metric(self, metric: Metric) -> None:  # noqa: D102
        self._metric = metric

    def get_metric(self) -> Metric:  # noqa: D102
        return self._metric

    @property
    def metric(self) -> Metric:  # noqa: D102
        return self.get_metric()


class Trainer:
    """Trainer class to handle the training and hyperparameter optimization of an estimator."""

    def __init__(self,
                 estimator: Estimator,
                 args: TrainingArguments,
                 tracker: Optional[ModelTracker] = None,
                 logger: Optional[Logger] = None):
        """Initialize the Trainer with an estimator, hyperparameters, and optional tracking and logging components."""
        # Catch component attributes (public)
        self.estimator = estimator
        self.args = args
        self.tracker = tracker
        self.logger = logger
        # Training configuration (public)
        self.metric = args.metric

        # Operation flags
        self._hp_optimized = False

    def train(self,
              X_train: Any,
              y_train: Optional[Any] = None, # Unsupervised learning, or dataloaders
              X_val: Optional[Any] = None,
              y_val: Optional[Any] = None,
              save_model_path: Optional[Union[str, pathlib.Path]] = None,
              debug: bool = False) -> None:
        """Train the estimator with the given data and hyperparameters."""
        # Get fixed hyperparameters from TrainingArguments
        hyperparameters = self.args.get_fixed_hyperparameters()
        validation = False
        if X_val is not None:
            validation = True

        # Check if we have performed hyperparameter optimization
        # If not, we need to pull the tunable hps and, take a random hp suggestion
        # and set them as parameters.
        # If optimization has been done, we should have the best hyperparameters
        # already fixed.
        if not self._hp_optimized:
            # Get tunable hyperparameters from TrainingArguments
            hyperparameters_tunable_sampled = self.args.sample_tunable_hyperparameter_space()

            # Combine fixed and tunable hyperparameters
            hyperparameters = {**hyperparameters, **hyperparameters_tunable_sampled}

        # Set parameters to the estimator
        self.estimator.set_params(**hyperparameters)

        if self.logger is not None:
            self.logger.info(f"\nHyperparameters set for training: {hyperparameters}")

        if self.tracker:
            self.tracker.start_run("Training Run")
            self.tracker.log_params(self.args.params)

        # Fit the estimator
        self.estimator.fit(X_train, y_train)

        if self.logger is not None:
            self.logger.info("Model successfully fit!")

        # Save the model if a path is provided
        if save_model_path is not None:
            self.save_estimator(save_model_path)
            if self.logger is not None:
                self.logger.info(f"Model saved locally to {save_model_path}")

        # Log metrics and model to the tracker, if provided
        metrics = None
        if self.tracker is not None:
            metrics = None
            if validation:
                metrics = {self.metric: self.estimator.evaluate(X_val, y_val, self.metric)}
            else:
                metrics = {self.metric: self.estimator.evaluate(X_train, y_train, self.metric)}
            if metrics is not None:
                for key, value in metrics.items():
                    self.tracker.log_metrics(key, value)
            if save_model_path is not None:
                self.tracker.log_artifact(local_path=save_model_path)

        if self.logger is not None and metrics is not None:
            self.logger.info(f"Model evaluated; metric with {'val' if validation else 'train'} split: {metrics}")

        if self.tracker is not None:
            self.tracker.end_run()

    def optimize_hyperparameters(self,
                                 X_train: Any,
                                 y_train: Optional[Any] = None, # Unsupervised learning or dataloaders
                                 X_val: Optional[Any] = None,
                                 y_val: Optional[Any] = None,
                                 n_trials: int = 60,
                                 debug: bool = False) -> None:
        """Optimize the hyperparameters of the estimator using Optuna."""
        def objective(trial):
            """Objective function to optimize the hyperparameters."""
            # Create hyperparameter space from TrainingArguments
            hyperparameters_tunable = self.args.get_tunable_hyperparameter_space(trial)

            # Get fixed hyperparameters
            hyperparameters_fixed = self.args.get_fixed_hyperparameters()

            # Set tunable and fixed hyperparameters to the estimator
            hyperparameters = {**hyperparameters_fixed, **hyperparameters_tunable}
            self.estimator.set_params(**hyperparameters)

            # Fit the estimator with tunable hyperparameters
            self.estimator.fit(X_train, y_train)

            # Evaluate the estimator
            if X_val is not None:
                score = self.estimator.evaluate(X_val, y_val, self.metric)
            else:
                score = self.estimator.evaluate(X_train, y_train, self.metric)

            return score

        # Run the optimization with Optuna
        direction = get_optimization_direction(self.metric)
        study = optuna.create_study(direction=direction.value)
        study.optimize(objective, n_trials=n_trials)

        # Update TrainingArguments with the best hyperparameters
        best_params = study.best_params
        if self.logger is not None:
            self.logger.info(f"Trainer.optimize_hyperparameters(): Hyperparameters optimized. Best set: {best_params}")
        self._hp_optimized = True
        self.args.update(best_params) # They become fixed hyperparameters, not tunable anymore

    def save_estimator(self, path: Union[str, pathlib.Path]) -> None:  # noqa: D102
        self.estimator.save_estimator(path)
        if self.logger is not None:
            self.logger.info(f"Trainer.save_estimator(): Saved estimator to {path}.")

    def load_estimator(self, path: Union[str, pathlib.Path]) -> None:  # noqa: D102
        self.estimator.load_estimator(path)
        if self.logger is not None:
            self.logger.info(f"Trainer.load_estimator(): Loaded estimator from {path}.")
