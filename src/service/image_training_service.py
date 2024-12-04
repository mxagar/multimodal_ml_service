"""This module contains the service functions for training the image processing pipelines.

Any domain specific use-case which has a module under src/domain/image
should have a corresponding reference here. Thus, this module is also a catalogue
of all the image processing pipelines used in the project.

An example of how to train the image processing pipelines provided in run_example().
However, the ideal entrypoint would be rather the function train_image_processing_pipelines()
called from an API, a GUI or a CLI.

Author: Mikel Sagardia
Date: 2024-12-04
"""
from typing import Optional, List

import numpy as np

from src.core import CONFIG_PATH, LOGS_PATH
from src.adapters.logger import Logger
from src.adapters.tracker import create_model_tracker
from src.domain.image import blur
from src.service.dataset_service import image_datasets


# This list must be synchronized with the function train_image_processing_pipelines()
ALL_IMAGE_TRAINING_PIPELINE_NAMES = [
    "blur_gradients",
    "blur_laplacian",
    "brightness"
]
# Modify this by selecting a subset of desired models from the list above
SELECTED_IMAGE_TRAINING_PIPELINE_NAMES = [
    "blur_gradients",
    "blur_laplacian"
]


def train_blur_model(model_type: str, X: np.ndarray | List[np.ndarray], y: str | int | List[str | int]) -> None:
    """Trains a blur detection model and saves the trained model to the specified path."""
    blur_estimator_yaml = CONFIG_PATH / f"blur_estimator_{model_type}.yaml"

    # Initialize the Logger
    logger = Logger(name="multimodal_service_logger_training",
                    log_file=LOGS_PATH / "multimodal_service_blur_training.log")

    # Initialize the ModelTracker
    # NOTE: The MLflow server must have been started before running scripts/start_mlflow_server.sh
    tracker = create_model_tracker(experiment_name="blur_detection", logger=logger)

    blur_training_pipeline = blur.create_blur_training_pipeline(
        model_type=model_type,
        estimator_yaml_path=blur_estimator_yaml,
        load=False,
        logger=logger,
        tracker=tracker,
    )
    blur_training_pipeline.run_training_pipeline(
        X_train=X, y_train=y, run_hp_optimization=True, split_size=0.2, debug=True
    )


def train_image_processing_pipelines(pipeline_names: Optional[List[str]] = None) -> None:
    """Train the image processing pipelines passed in the arguments."""
    if pipeline_names is None:
        pipeline_names = ALL_IMAGE_TRAINING_PIPELINE_NAMES

    # Dataset
    blur_dataset = image_datasets.blur
    blur_dataset.load_images()
    images = blur_dataset.images
    labels = blur_dataset.labels

    if "blur_gradients" in pipeline_names:
        train_blur_model("gradients", X=images, y=labels)
    if "blur_laplacian" in pipeline_names:
        train_blur_model("laplacian", X=images, y=labels)
    if "brightness" in pipeline_names:
        # TODO: We can extend this to include other image processing pipelines
        pass


def run_example() -> None:
    """Run an example of training the image processing pipelines."""
    train_image_processing_pipelines(pipeline_names=SELECTED_IMAGE_TRAINING_PIPELINE_NAMES)


if __name__ == "__main__":
    run_example()
