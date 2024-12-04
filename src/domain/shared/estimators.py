"""Estimators module.

This module contains the abstract base classes for all estimators,
both rule-based and machine learning models:

- RuleBasedModel: Abstract base class for all rule-based models.
- Estimator: Abstract base class for all estimators, both rule-based and machine learning models.
- SklearnPipeEstimator: An Estimator implementation that wraps around an sklearn Pipeline.
- RuleBasedEstimator: An Estimator implementation that wraps around a rule-based model.
- PytorchEstimator: To be done...
- YOLOEstimator: To be done...
- ONNXEstimator: To be done...
- LLMEstimator: To be done...

An Estimator is a wrapper class for a model, which can be rule-based or a machine learning model.
It provides methods for:

- fitting
- setting and getting parameters,
- saving and loading parameters,
- predicting,
- and loading and saving the underlying model.

Author: Mikel Sagardia
Date: 2024-12-04
"""
from abc import ABC, abstractmethod
import pathlib
from typing import Any, Dict, List, Union, Optional
import joblib
import yaml

from sklearn.pipeline import Pipeline

from src.domain.shared.evaluation import Metric, Evaluator


class RuleBasedModel(ABC):
    """Abstract base class for all rule-based models.

    Every rule-based model should inherit from this class
    and implement the following methods:
    - set_params: set parameters
    - get_params: get parameters
    - predict: make predictions

    Note that the definition of fit() is necessary for compliance with the Estimator class
    and the Trainer, but it is not used in most rule-based models.
    Instead, the parameters are tuned by the Trainer.
    Basically, the rule-based models are defined by algorithmic rules
    which occur inside the predict() method; those algorithmic
    rules make use of parameter values, which should be attributes of the derived class.
    """

    def __init__(self):
        super().__init__()

    # Implement this method in the derived class, if necessary
    def fit(self, X: Any, y: Any) -> "RuleBasedModel":  # noqa: D102
        return self

    @abstractmethod
    # Implement this method in the derived class
    def set_params(self, **params: Any) -> None:  # noqa: D102
        pass

    @abstractmethod
    # Implement this method in the derived class
    def get_params(self) -> dict:  # noqa: D102
        pass

    @abstractmethod
    # Implement this method in the derived class
    def predict(self, X: Any) -> Any:  # noqa: D102
        # The algorithmic logic needs to be implemented here
        # It should make use of the parameters set by set_params
        # which will be the attributes of the class
        pass


class Estimator(ABC):
    """Abstract base class for all estimators, both rule-based and machine learning models."""

    def __init__(self):
        super().__init__()
        # Default metric of Evaluator is F1,
        # but it can be redefined every call to method evaluate()
        # and it persists as attribute of Evaluator instance
        self.evaluator = Evaluator()

    @abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None) -> "Estimator":  # noqa: D102
        # y is optional to handle unsupervised learning or dataloaders (Pytorch)
        return self

    @abstractmethod
    def set_params(self, **params: Any) -> None:  # noqa: D102
        pass

    @abstractmethod
    def get_params(self) -> dict:  # noqa: D102
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:  # noqa: D102
        pass

    @abstractmethod
    def load_estimator(self, path: Union[str, pathlib.Path]) -> None:  # noqa: D102
        pass

    @abstractmethod
    def save_estimator(self, path: Union[str, pathlib.Path]) -> None:  # noqa: D102
        pass

    def evaluate(self, X: Any, y: Optional[Any] = None, metric: Optional[Metric] = None) -> Any:  # noqa: D102
        # y is optional to handle unsupervised learning or dataloaders (Pytorch)
        # The Evaluator always has a default metric, but it can be redefined here
        y_pred = self.predict(X)
        result = self.evaluator.evaluate(y, y_pred, metric)
        return result


class SklearnPipeEstimator(Estimator):
    """An Estimator implementation that wraps around an sklearn Pipeline.

    It can be fit() on its own or also with a Trainer.

    Usage example:

        ...
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier())
        ])

        pipeline_estimator = SklearnPipeEstimator(pipe)
        pipeline_estimator = get_pipeline_steps() # ['scaler', 'xgb']

        params = {
            'xgb': {
                'n_estimators': 100,
                'learning_rate': 0.1
            }
        }
        pipeline_estimator.set_params(nested=True, **params)
        params_ = pipeline_estimator.get_params(nested=True) # same as params

        pipeline_estimator.fit(X_train, y_train)
        metric = pipeline_estimator.evaluate(X_test, y_test, metric=Metric.ACCURACY)
        pred = pipeline_estimator.predict(X_test)

    """

    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.pipeline = pipeline

    def fit(self, X: Any, y: Optional[Any] = None) -> "SklearnPipeEstimator": # noqa: D102
        # y is optional to handle unsupervised learning or dataloaders
        self.pipeline.fit(X, y)
        return self

    def set_params(self, nested: bool = False, **params: Any) -> None: # noqa: D102
        if nested:
            for step_name, step_params in params.items():
                for param_name, value in step_params.items():
                    self.pipeline.set_params(**{f"{step_name}__{param_name}": value})
        else:
            self.pipeline.set_params(**params)

    def get_params(self, nested: bool = False) -> Dict[str, Any]: # noqa: D102
        params = self.pipeline.get_params()
        if nested:
            formatted_params = {}
            for key, value in params.items():
                if "__" in key:
                    step_name, param_name = key.split("__", 1)
                    if step_name not in formatted_params:
                        formatted_params[step_name] = {}
                    formatted_params[step_name][param_name] = value
            params = formatted_params

        return params

    def predict(self, X: Any) -> Any: # noqa: D102
        return self.pipeline.predict(X)

    def load_estimator(self, path: Union[str, pathlib.Path]) -> None: # noqa: D102
        self.pipeline = joblib.load(path)

    def save_estimator(self, path: Union[str, pathlib.Path]) -> None: # noqa: D102
        joblib.dump(self.pipeline, path)

    def get_pipeline_steps(self) -> List[str]: # noqa: D102
        return list(self.pipeline.named_steps.keys())


class RuleBasedEstimator(Estimator):
    """An Estimator implementation that wraps around a rule-based model.

    It can be fit() on its own or also with a Trainer.

    Usage example:

        class MyRuleBasedModel(RuleBasedModel):
            def __init__(self, threshold: float = 0.5):
                self.threshold = threshold
                super().__init__()

            def set_params(self, **params: Any) -> None:
                for key, value in params.items():
                    setattr(self, key, value)

            def get_params(self) -> Dict[str, Any]:
                return {"threshold": self.threshold}

            def predict(self, X: Any) -> Any:
                # Example rule-based prediction logic
                return [1 if x > self.threshold else 0 for x in X]

        model = MyRuleBasedModel()
        estimator = RuleBasedEstimator(model)

        estimator.set_params(threshold=0.7)
        params = estimator.get_params() # {'threshold': 0.7}

        estimator.fit(X_train, y_train) # Nothing happens usually
        pred = estimator.predict(X_test) # Internal algorithm is run with params

        estimator.save_estimator('model_params.yaml')
        estimator.load_estimator('model_params.yaml')
    """
    def __init__(self, model: RuleBasedModel):
        super().__init__()
        self.model = model

    def fit(self, X: Any, y: Optional[Any] = None) -> "RuleBasedEstimator": # noqa: D102
         # y is optional to handle unsupervised learning or dataloaders
        self.model.fit(X, y)
        return self

    def set_params(self, **params: Any) -> None: # noqa: D102
        self.model.set_params(**params)

    def get_params(self) -> Dict[str, Any]: # noqa: D102
        return self.model.get_params()

    def predict(self, X: Any) -> Any: # noqa: D102
        return self.model.predict(X)

    def load_estimator(self, path: Union[str, pathlib.Path]) -> None: # noqa: D102
        with open(path, "r") as file:
            params = yaml.safe_load(file)
        self.set_params(**params)

    def save_estimator(self, path: Union[str, pathlib.Path]) -> None: # noqa: D102
        params = self.get_params()
        with open(path, "w") as file:
            yaml.dump(params, file)


class PytorchEstimator(Estimator):  # noqa: D101
    # NOTE: We should get dataloaders instead of loaded features and targets (X, y)
    # That is: X can be a dataloader and y is set to None, or not passed at all
    # while we internally only use the dataloader to get both X and y
    pass


class YOLOEstimator(Estimator):  # noqa: D101
    pass


class ONNXEstimator(Estimator):  # noqa: D101
    pass


class LLMEstimator(Estimator):  # noqa: D101
    pass
