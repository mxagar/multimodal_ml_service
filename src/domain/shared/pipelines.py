"""Training and inference pipelines to train and evaluate an estimator.

Contents:

- TrainingPipeline: Pipeline to train and evaluate an estimator. Its entry-point is run_training_pipeline(),
which orchestrates the training process, including hyperparameter optimization, data transformation,
and model evaluation. Finally, it saves the model and data transformers.
- InferencePipeline: Pipeline to transform and predict using an estimator. Its entry-point is run_inference_pipeline(),
which transforms the input data and predicts the output using the estimator.

The tracking (via MLflow) is delegated to the Trainer class, which is used in the TrainingPipeline.

Note that both pipelines support

- any series of data transformers, which are used to preprocess the data before training and inference
- and any estimator, which is used to train and predict.

If we wish to train a data transformer, though, we should defined it inside a SklearnPipeEstimator, i.e.,
not in the data transformers list of the pipeline. The SklearnPipeEstimator is a custom estimator
which contains an sklearn Pipeline, which can include preprocessing steps and a model at the end.
Since our data transformers are derived from sklearn transformers, they can be included in that
sklearn Pipeline, too.

TODO: To avoid code duplication, we could refactor the TrainingPipeline and InferencePipeline classes
to inherit from a common base class, e.g., OperationPipeline,
which would contain the shared methods and attributes.

Author: Mikel Sagardia
Date: 2024-12-04
"""
import pathlib
from typing import Any, Union, Optional, List

import numpy as np
from sklearn.model_selection import train_test_split

from src.domain.shared.estimators import Estimator
from src.domain.shared.training import Trainer, TrainingArguments
from src.adapters.tracker import ModelTracker
from src.adapters.logger import Logger
from src.domain.shared.data import (DataTransformer,
                                    save_data_transformers)


class TrainingPipeline:
    """Training pipeline to train and evaluate an estimator."""

    def __init__(self,
                 estimator: Estimator,
                 args: TrainingArguments,
                 save_model_path: Optional[Union[str, pathlib.Path]] = None,
                 transformers: Optional[List[DataTransformer]] = None,
                 save_transformers_path: Optional[Union[str, pathlib.Path]] = None,
                 tracker: Optional[ModelTracker] = None,
                 logger: Optional[Logger] = None) -> None:
        """Initialize the training pipeline."""
        self.estimator = estimator
        self.args = args
        self.save_model_path = save_model_path
        self.transformers = transformers
        self.save_transformers_path = save_transformers_path
        self.tracker = tracker
        self.logger = logger
        self.trainer = Trainer(estimator=estimator, args=args, tracker=tracker)
        if self.logger is not None:
            self.logger.info("TrainingPipeline: Initialized.")

    def run_training_pipeline(self,
                              X_train: Any,
                              y_train: Optional[Any] = None, # optional, to allow unsupervised learning and dataloaders
                              X_val: Optional[Any] = None,
                              y_val: Optional[Any] = None,
                              X_test: Optional[Any] = None,
                              y_test: Optional[Any] = None,
                              run_hp_optimization: bool = True,
                              split_size: float = 0.2,
                              debug: bool = False) -> None:
        """Run the training pipeline; main entry-point."""
        # Create splits
        # 1. TEST: if not passed, created from TRAIN
        # 2. VAL: if not passed, created from TRAIN remaining after splitting TEST
        if X_test is None:
            if y_train is None:
                X_train, X_test = train_test_split(X_train, test_size=split_size, random_state=42)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                                    y_train,
                                                                    test_size=split_size,
                                                                    random_state=42)
        if X_val is None:
            if y_train is None:
                X_train, X_val = train_test_split(X_train, test_size=split_size, random_state=42)
            else:
                X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                                  y_train,
                                                                  test_size=split_size,
                                                                  random_state=42)

        # Fit transformers with train split
        # (usually no fitting is done, but just in case...)
        if self.transformers is not None:
            self.fit_transformers(X_train, y_train)

        # Transform data
        if self.transformers is not None:
            # Transform all splits
            X_train, y_train = self.transform_data(X_train, y_train)
            X_val, y_val = self.transform_data(X_val, y_val)
            X_test, y_test = self.transform_data(X_test, y_test)

        # Train: optionally, optimize hyperparameters first
        self.train(X_train, y_train,
                   X_val, y_val,
                   run_hp_optimization=run_hp_optimization,
                   debug=debug)

        # Evaluate on test split
        result = self.evaluate(X_test, y_test)
        # FIXME: we should log the result to the tracker, but the run is closed...?
        if self.logger is not None:
            self.logger.info(
                f"TrainingPipeline.run_training_pipeline(): Evaluation metrics on test split: {self.args.metric.value} = {result}")  # noqa: E501

        # Save model
        # This is happens in the Trainer: if we a valid save_model_path, it will save the model
        # and if it has a valid ModelTracker, the model will be logged to the tracker artifactory
        # Why so? Because the Trainer should be able to save the fit model and log it to the tracker
        # In contrast, the transformers used in the pipeline are preprocessing steps which are not tuned.
        # The transformers which are tuned should be defined as part of the estimator,
        # e.g., in a SklearnPipeEstimator.

        # Save data transformers
        if self.transformers is not None and self.save_transformers_path is not None:
            save_data_transformers(self.transformers, self.save_transformers_path)

    def fit_transformers(self, X: Any, y: Optional[Any] = None) -> None:
        """Fit the transformers on the data."""
        if self.transformers is not None:
            if self.logger is not None:
                self.logger.info(
                    f"TrainingPipeline.fit_transformers(): Fitting {len(self.transformers)} transformers...")
            for t in self.transformers:
                t = t.fit(X, y)  # noqa: PLW2901
            if self.logger is not None:
                self.logger.info("TrainingPipeline.fit_transformers(): Transformers prepared.")
        elif self.logger is not None:
            self.logger.warning("TrainingPipeline.fit_transformers(): No transformers set.")

    def transform_data(self, X: Any, y: Optional[Any] = None) -> Union[List[Any], np.ndarray]:
        """Transform the data using the transformers."""
        if self.transformers is not None:
            if self.logger is not None:
                self.logger.info("TrainingPipeline.transform_data(): Starting data transformation...")
            Xt = X
            yt = y
            for t in self.transformers:
                Xt = t.transform(Xt, yt)
                if isinstance(Xt, tuple) and len(Xt) == 2: # (X, y)  # noqa: PLR2004
                    yt = Xt[1] # pick y
                    Xt = Xt[0] # pick X
            if self.logger is not None:
                self.logger.info("TrainingPipeline.transform_data(): Data transformed.")
            return Xt, yt
        else:
            if self.logger is not None:
                self.logger.warning("TrainingPipeline.transform_data(): No transformers set.")
            return Xt, yt

    def train(self,
              X_train: Any, y_train: Optional[Any] = None,
              X_val: Optional[Any] = None, y_val: Optional[Any] = None,
              run_hp_optimization: bool = True,
              debug: bool = False) -> None:
        """Run the training process; hyperparameter optimization can be performed before estimator training."""
        if run_hp_optimization:
            if self.logger is not None:
                self.logger.info("TrainingPipeline.train(): Starting hyperparameter optimization...")
            self.trainer.optimize_hyperparameters(
                X_train,
                y_train,
                X_val,
                y_val,
                n_trials=60,
                debug=debug
            )
        if self.logger is not None:
            self.logger.info("TrainingPipeline.train(): Starting Estimator training...")
        self.trainer.train(X_train, y_train, debug=debug, save_model_path=self.save_model_path)
        if self.logger is not None:
            self.logger.info("TrainingPipeline.train(): Estimator trained.")

    def evaluate(self, X: Any, y: Optional[Any] = None) -> Any:
        """Evaluate the estimator on the given data."""
        if self.logger is not None:
            self.logger.info("TrainingPipeline.evaluate(): Starting Estimator evaluation...")
        result = self.estimator.evaluate(X, y, metric=self.args.metric)
        if self.logger is not None:
            self.logger.info(f"TrainingPipeline.evaluate(): result = {result}.")
        return result

    def set_estimator(self, estimator: Estimator) -> None:  # noqa: D102
        self.estimator = estimator

    def get_estimator(self) -> Estimator:  # noqa: D102
        return self.estimator

    def save_estimator(self, path: Union[str, pathlib.Path]) -> None:  # noqa: D102
        self.estimator.save_estimator(path)
        if self.logger is not None:
            self.logger.info(f"TrainingPipeline.save_estimator(): Saved estimator to {path}.")

    def load_estimator(self, path: Union[str, pathlib.Path]) -> None:  # noqa: D102
        self.estimator.load_estimator(path)
        if self.logger is not None:
            self.logger.info(f"TrainingPipeline.load_estimator(): Loaded estimator from {path}.")

    def set_transformers(self, transformers: Union[DataTransformer, List[DataTransformer]]) -> None:  # noqa: D102
        if isinstance(transformers, DataTransformer):
            transformers = [transformers]
        self.transformers = transformers
        if self.logger is not None:
            self.logger.warning("TrainingPipeline.set_transformers(): Data transformers set.")

    def get_transformers(self) -> List[DataTransformer]:  # noqa: D102
        return self.transformers

    def save_transformers(self, path: Union[str, pathlib.Path, List[str], List[pathlib.Path]]) -> None:  # noqa: D102
        if not isinstance(path, list):
            path = [path]
        if len(path) != len(self.transformers):
            raise ValueError("TrainingPipeline.save_transformers(): Number of paths must match number of transformers.")
        for t, p in zip(self.transformers, path):
            t.save_transformer(p)

    def load_transformers(self, path: Union[str, pathlib.Path, List[str], List[pathlib.Path]]) -> None:  # noqa: D102
        if not isinstance(path, list):
            path = [path]
        if len(path) != len(self.transformers):
            raise ValueError("TrainingPipeline.load_transformers(): Number of paths must match number of transformers.")
        for t, p in zip(self.transformers, path):
            t.load_transformer(p)

    def convert_to_inference_pipeline(self) -> "InferencePipeline":
        """Convert the training pipeline to an inference pipeline."""
        return InferencePipeline(estimator=self.estimator,
                                 transformers=self.transformers,
                                 logger=self.logger)


class InferencePipeline:
    """Inference pipeline to transform and predict using an estimator."""

    def __init__(self,
                 estimator: Optional[Estimator] = None,
                 transformers: Optional[List[DataTransformer]] = None,
                 logger: Optional[ModelTracker] = None) -> None:
        """Initialize the inference pipeline."""
        self.estimator = estimator
        self.transformers = transformers
        self.logger = logger
        if self.logger is not None:
            self.logger.info("InferencePipeline: Initialized.")
        if self.logger is not None and self.estimator is None:
            self.logger.warning("InferencePipeline: No estimator set yet.")

    def run_inference_pipeline(self, X: Any, y: Optional[Any] = None) -> Any:
        """Run the inference pipeline; main entry-point."""
        # Transform
        Xt = X
        yt = y
        if self.transformers is not None:
            Xt = self.transform_data(Xt, yt)
            if isinstance(Xt, tuple) and len(Xt) == 2: # (X, y)  # noqa: PLR2004
                yt = Xt[1] # pick y
                Xt = Xt[0] # pick X
        # Predict
        pred = self.predict(Xt)

        return pred

    def predict(self, X: Any) -> Any:
        """Predict the output based on the input data."""
        pred = None
        if self.estimator is not None:
            pred = self.estimator.predict(X)
        else:
            if self.logger is not None:
                self.logger.info("InferencePipeline.predict(): Missing estimator.")
            raise ValueError("InferencePipeline.predict(): Missing estimator.")

        return pred

    def transform_data(self, X: Any, y: Optional[Any] = None) -> Union[List[Any], np.ndarray]:
        """Transform the data using the transformers."""
        if self.transformers is not None:
            if self.logger is not None:
                self.logger.info("InferencePipeline.transform_data(): Starting data transformation...")
            Xt = X
            yt = y
            for t in self.transformers:
                Xt = t.transform(Xt, yt)
                if isinstance(Xt, tuple) and len(Xt) == 2: # (X, y)  # noqa: PLR2004
                    yt = Xt[1] # pick y
                    Xt = Xt[0] # pick X
            if self.logger is not None:
                self.logger.info("InferencePipeline.transform_data(): Data transformed.")
            return Xt, yt
        else:
            if self.logger is not None:
                self.logger.warning("InferencePipeline.transform_data(): No transformers set.")
            return Xt, yt

    def set_estimator(self, estimator: Estimator) -> None:  # noqa: D102
        self.estimator = estimator
        if self.logger is not None and self.estimator is None:
            self.logger.warning("InferencePipeline.set_estimator(): Estimator set.")

    def get_estimator(self) -> Estimator:  # noqa: D102
        return self.estimator

    def load_estimator(self, path: Union[str, pathlib.Path]) -> None:  # noqa: D102
        self.estimator.load_estimator(path)
        if self.logger is not None:
            self.logger.info(f"InferencePipeline.load_estimator(): Loaded estimator from {path}.")

    def set_transformers(self, transformers: Union[DataTransformer, List[DataTransformer]]) -> None:  # noqa: D102
        if isinstance(transformers, DataTransformer):
            transformers = [transformers]
        self.transformers = transformers
        if self.logger is not None:
            self.logger.warning("InferencePipeline.set_transformers(): Data transformers set.")

    def get_transformers(self) -> List[DataTransformer]:  # noqa: D102
        return self.transformers

    def load_transformers(self, path: Union[str, pathlib.Path, List[str], List[pathlib.Path]]) -> None:  # noqa: D102
        if not isinstance(path, list):
            path = [path]
        if len(path) != len(self.transformers):
            raise ValueError("InferencePipeline.load_transformers(): Num. of paths must match num. of transformers.")
        for t, p in zip(self.transformers, path):
            t.load_transformer(p)
