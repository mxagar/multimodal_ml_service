"""Example domain module for blur detection using edge detection filters.

This module uses the modules in src/domain/shared
and defined specific classes and functions to train and use a blur detection model:

- extract_laplacian_from_image: A function that computes the Laplacian of an image.
- extract_gradients_from_image: A function that extracts features from an image using edge detection filters.
- GradientExtractor: A data transformer that extracts gradient features from an image.
- LaplacianBlurModel: A rule-based model that uses the variance of the Laplacian to determine if an image is blurry.
- create_blur_estimator: A factory function to create a blur estimator based on the model type.
- create_blur_transformers: A factory function to create blur transformers based on the model type.
- create_blur_training_pipeline: A factory function to create a blur training pipeline based on the model type.
- create_blur_inference_pipeline: A factory function to create a blur inference pipeline based on the model type.

An example is provided to demonstrate how to train and use a blur detection model at run_example().
However, this module is not intended to be run as a script; instead, it should be imported and used in

- src/service/image_training_service.py
- src/service/image_inference_service.py
- src/entrypoints/api.py

TODO: The factories and classes in this module could be refactored and abstracted to a more general
form to be used in other domains, such as text, audio, etc.

Author: Mikel Sagardia
Date: 2024-12-04
"""
from typing import Union, List, Any, Optional
import pathlib
from PIL import Image

import cv2
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config.config import (EstimatorConfig,
                               load_config_yaml)
from src.core import (DATA_PATH,
                      ARTIFACTS_PATH,
                      CONFIG_PATH,
                      LOGS_PATH)
from src.adapters.tracker import create_model_tracker, ModelTracker
from src.adapters.logger import Logger
from src.domain.shared.data import (DataTransformer,
                                    ImageDataset,
                                    load_data_transformers)
from src.domain.shared.estimators import (RuleBasedModel,
                                          Estimator,
                                          SklearnPipeEstimator,
                                          RuleBasedEstimator)
from src.domain.shared.training import TrainingArguments
from src.domain.shared.evaluation import (get_metric_from_string)
from src.domain.shared.pipelines import TrainingPipeline, InferencePipeline
from src.domain.shared.image_utils import (ChannelOrder,
                                           ImageFormat)


LAPLACE_FEATURE_NAMES = ["laplacian_var"]
GRADIENT_FEATURE_NAMES = [
    "sobel_mean", "sobel_max", "sobel_var", "laplacian_mean", "laplacian_max", "laplacian_var"
]


def extract_gradients_from_image(image: Union[Image.Image, np.ndarray],
                                 kernel_size_sobel: int = 3) -> List[float]:
    """Extract features from an image using various edge detection filters.

    This function converts the input image to grayscale and applies Sobel
    and Laplacian edge detection filters. It then calculates and returns the mean,
    maximum, and variance of the edges detected by each filter.

    Args:
    ----
        image (numpy.ndarray): The input image in RGB format.
        kernel_size_sobel (int): The size of the kernel to use for the Sobel filter.

    Returns:
    -------
        list: A list of features containing the mean, maximum, and variance of the
              edges detected by Sobel and Laplacian filters.

    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply edge detection filters using OpenCV
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size_sobel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size_sobel)
    sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
    laplacian_edges = cv2.Laplacian(gray, cv2.CV_64F)

    # Calculate features for each edge detection filter
    features = []
    for edges in [sobel_edges, laplacian_edges]:
        f = [np.mean(edges), np.max(edges), np.var(edges)]
        features.extend(f)

    return features


def extract_laplacian_from_image(image: Union[Image.Image, np.ndarray]) -> float:
    """Compute the Laplacian.

    Args:
    ----
        image (numpy.ndarray): The input image in RGB format.

    Returns:
    -------
        float: The variance of the Laplacian of the image.

    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Compute Laplacian
    feature = cv2.Laplacian(gray, cv2.CV_64F).var()

    return feature


class LaplacianExtractor(DataTransformer):  # noqa: D101
    # We could implement this, but not is not really necessary
    # because we can use use the extract_laplacian_from_image() function
    # in the RuleBasedModel class derived below.
    pass


class GradientExtractor(DataTransformer):
    """Data transformer that extracts gradient features from images."""
    def __init__(self,
                 kernel_size_sobel: int = 3,
                 feature_names: List[str] = GRADIENT_FEATURE_NAMES) -> None:
        super().__init__()
        self._kernel_size_sobel = kernel_size_sobel
        self._feature_names = feature_names

    def transform(self,
                  #X: Union[Any, List[Any]],
                  X: Union[Union[Image.Image, np.ndarray], List[Union[Image.Image, np.ndarray]]],
                  y: Any = None) -> np.ndarray:
        """Transform the input image(s) to an array of gradient statistics."""
        image_features = None
        if isinstance(X, List):
            image_features = [extract_gradients_from_image(image,
                                                           kernel_size_sobel=self._kernel_size_sobel) for image in X]
        else:
            image_features = extract_gradients_from_image(X,
                                                          kernel_size_sobel=self._kernel_size_sobel)
            image_features = np.expand_dims(image_features, axis=0)

        # size (n, f) if list of n images with f features each
        # size (1, f) if single image
        return np.array(image_features)

    def get_params(self) -> dict:  # noqa: D102
        return {"kernel_size_sobel": self._kernel_size_sobel,
                "feature_names": self._feature_names}

    def set_params(self, **params: Any) -> None:  # noqa: D102
        if "kernel_size_sobel" in params:
            self._kernel_size_sobel = params["kernel_size_sobel"]
        if "feature_names" in params:
            self._feature_names = params["feature_names"]


class LaplacianBlurModel(RuleBasedModel):
    """Rule-based model that uses the image var(Laplacian) to state whether the image is blurry or not."""

    def __init__(self, threshold: int = 100):
        super().__init__()
        self._threshold = threshold

    def set_params(self, **params: Any) -> None:  # noqa: D102
        if "threshold" in params:
            self._threshold = params["threshold"]

    def get_params(self) -> dict:  # noqa: D102
        return {"threshold": self._threshold}

    def predict(self, X: Union[List, Union[Image.Image, np.ndarray]]) -> np.ndarray:  # noqa: D102
        pred = None
        if isinstance(X, list):
            # The rule-based model is applied to each image in the list here,
            # i.e., in this case, the unique consists rule in checking the threshold value
            pred = [int(extract_laplacian_from_image(x) < self._threshold) for x in X]
        else:
            pred = int(extract_laplacian_from_image(X) < self._threshold)
            pred = np.expand_dims(pred, axis=0)

        # size (n, 1) if list of n images with a unique output from model each
        # size (1, 1) if single image
        return np.array(pred)


def create_blur_estimator(model_type: str,
                          estimator_path: Optional[Union[str, pathlib.Path]] = None,
                          load: bool = False,
                          logger: Optional[Logger] = None) -> Estimator:
    """Factory function to create a blur estimator based on the model type."""
    estimator = None
    if model_type == "laplacian":
        # RuleBasedEstimator <- (LaplacianExtractor), LaplacianBlurModel  # noqa: ERA001
        model = LaplacianBlurModel()
        estimator = RuleBasedEstimator(model)
    elif model_type == "gradients":
        # SklearnPipeEstimator <- Pipeline((GradientExtractor), StandardScaler, XGBoostClassifier) # noqa: ERA001
        pipe = Pipeline([
            #('gradients', GradientExtractor()), # noqa: ERA001
            ("scaler", StandardScaler()),
            ("xgb", XGBClassifier())
        ])
        estimator = SklearnPipeEstimator(pipe)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    if load and estimator_path is not None:
        # Load parameters, weights, etc. from a file
        estimator.load_estimator(estimator_path)

    return estimator

def create_blur_transformers(model_type: str,
                             transformers_path: Optional[Union[str, pathlib.Path]] = None,
                             load: bool = False,
                             logger: Optional[Logger] = None) -> Union[List[DataTransformer], None]:
    """Factory function to create blur transformers based on the model type."""
    transformers = None # if "laplacian"
    if model_type == "laplacian":
        transformers = None
        #transformers = [LaplacianExtractor()]  # noqa: ERA001
    elif model_type == "gradients":
        #transformers = None  # noqa: ERA001
        transformers = [GradientExtractor()]
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    if load and transformers is not None and transformers_path is not None:
        # Set the parameters of the transformers from the YAML file
        load_data_transformers(transformers, transformers_path, logger=logger)

    return transformers


def create_blur_training_pipeline(model_type: str,
                                  estimator_yaml_path: Optional[Union[str, pathlib.Path]] = None,
                                  load: bool = False,
                                  logger: Optional[Logger] = None,
                                  tracker: Optional[ModelTracker] = None) -> TrainingPipeline:
    """Factory function to create a blur training pipeline based on the model type."""
    # Open Estimator YAML and extract parameters
    estimator_config_dict = load_config_yaml(estimator_yaml_path)
    estimator_config = EstimatorConfig(**estimator_config_dict)
    nested = estimator_config.hyperparameters.nested
    params = estimator_config.hyperparameters.values
    metric = get_metric_from_string(estimator_config.metric)
    save_model_path = None
    save_transformers_path = None
    if estimator_config.modelpath is not None:
        save_model_path = ARTIFACTS_PATH / estimator_config.modelpath
    if estimator_config.transformerspath is not None:
        save_transformers_path = ARTIFACTS_PATH / estimator_config.transformerspath

    # Estimator
    blur_estimator = create_blur_estimator(model_type=model_type,
                                           estimator_path=None, # save_model_path
                                           load=load) # instantiate from scratch, don't load params from YAML yet

    # Set TrainingArguments
    args = TrainingArguments(metric=metric, nested=nested, **params)

    # Instantiate transformers
    blur_transformers = create_blur_transformers(model_type=model_type,
                                                 transformers_path=None, # save_transformers_path
                                                 load=load) # instantiate from scratch, don't load params from YAML yet

    # Training pipeline
    blur_training = TrainingPipeline(estimator=blur_estimator,
                                     args=args,
                                     save_model_path=save_model_path,
                                     transformers=blur_transformers,
                                     save_transformers_path=save_transformers_path,
                                     tracker=tracker,
                                     logger=logger)

    return blur_training


def create_blur_inference_pipeline(model_type: str,
                                   estimator_yaml_path: Optional[Union[str, pathlib.Path]] = None,
                                   load: bool = True,
                                   logger: Optional[Logger] = None) -> InferencePipeline:
    """Factory function to create a blur inference pipeline based on the model type."""
    # Open Estimator YAML and extract parameters
    estimator_config_dict = load_config_yaml(estimator_yaml_path)
    estimator_config = EstimatorConfig(**estimator_config_dict)
    model_path = None
    transformers_path = None
    if estimator_config.modelpath is not None:
        model_path = ARTIFACTS_PATH / estimator_config.modelpath
    if estimator_config.transformerspath is not None:
        transformers_path = ARTIFACTS_PATH / estimator_config.transformerspath

    # Estimator
    blur_estimator = create_blur_estimator(model_type=model_type,
                                           estimator_path=model_path,
                                           load=load) # load parameters from file

    # Transformers
    blur_transformers = create_blur_transformers(model_type=model_type,
                                                 transformers_path=transformers_path,
                                                 load=load) # load parameters from file

    # Inference pipeline
    blur_inference = InferencePipeline(estimator=blur_estimator,
                                       transformers=blur_transformers,
                                       logger=logger)

    return blur_inference


def run_example(model_type: str = "gradients") -> None:
    """Run an example of training and inference using the blur detection model."""
    ### -- Configuration

    blur_dataset_yaml = CONFIG_PATH / "blur_dataset_kaggle.yaml"
    blur_estimator_yaml = CONFIG_PATH / f"blur_estimator_{model_type}.yaml"

    # Initialize the Logger
    logger = Logger(name="multimodal_service_logger_training",
                    log_file=LOGS_PATH / "multimodal_service_blur_training.log")

    # Initialize the ModelTracker
    # NOTE: The MLflow server must have been started before running scripts/start_mlflow_server.sh
    tracker = create_model_tracker(experiment_name="blur_training", logger=logger)

    ### -- Dataset

    #transformers = [ImageResizer(min_size=200, keep_aspect_ratio=True)]  # noqa: ERA001
    transformers = None
    ds = ImageDataset(
        yaml_path=blur_dataset_yaml,
        load_images=False,
        image_format=ImageFormat.NUMPY,
        channels=ChannelOrder.RGB,
        transformers=transformers
    )
    class_names = ds.class_names

    ds.load_images() # Load all images into memory
    images = ds.images # Get all images as a list (np.ndarray or PIL.Image)
    labels = ds.labels # Get all labels as a list

    logger.info("Loaded training images and labels.")
    logger.info(f"Number of images: {len(images)}")
    logger.info(f"Number labels: {len(labels)}")

    ### -- Training

    training_pipeline = create_blur_training_pipeline(model_type=model_type,
                                                      estimator_yaml_path=blur_estimator_yaml,
                                                      load=False,
                                                      logger=logger,
                                                      tracker=tracker)

    # Train: inside the pipeline:
    # - splitting is done
    # - data is transformed
    # - HP optimization is run
    # - model is trained
    # - model is persisted if a local path is provided in pipeline instantiation
    # - transformers are persisted if a local path is provided in pipeline instantiation
    training_pipeline.run_training_pipeline(X_train=images,
                                            y_train=labels,
                                            run_hp_optimization=True,
                                            split_size=0.2,
                                            debug=True)

    ### -- Inference

    logger = Logger(name="multimodal_service_logger_inference",
                    log_file=LOGS_PATH / "multimodal_service_blur_inference.log")

    # If we set load=True, the pipeline will load the model and transformers from the paths
    # provided in the Estimator YAML file
    inference_pipeline = create_blur_inference_pipeline(model_type=model_type,
                                                        estimator_yaml_path=blur_estimator_yaml,
                                                        load=True,
                                                        logger=logger)

    # Load an image with the same transformations used during the training
    image_path = DATA_PATH / pathlib.Path("kaggle_kwentar_blur_dataset/blur_dataset_resized_300/defocused_blurred/0_IPHONE-SE_F.JPG")  # noqa: E501
    img = ImageDataset.load_and_prepare_image(image_path, transformers=transformers)

    # Predict
    pred = inference_pipeline.run_inference_pipeline(img)
    logger.info(f"Image prediction: {class_names[pred[0]]}")

if __name__ == "__main__":
    run_example(model_type="laplacian")
    run_example(model_type="gradients")
