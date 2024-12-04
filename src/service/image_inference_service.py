"""This module contains the service functions for image inference.

It makes sense to use this module after training the image processing pipelines
has been carried out, i.e., via the module image_training_service.py.

Any domain specific use-case which has a module under src/domain/image
should have a corresponding reference here. Thus, this module is also a catalogue
of all the image processing pipelines used in the project.

An example of how to train the image processing pipelines provided in run_example().
However, the ideal entrypoint is the API, located in src/entrypoints/api.py.

Author: Mikel Sagardia
Date: 2024-12-04
"""
from typing import Union, List, Optional
import pathlib

from PIL import Image
import numpy as np

from src.core import CONFIG_PATH, DATA_PATH, LOGS_PATH
from src.adapters.logger import Logger
from src.domain.image import blur
from src.domain.shared.image_utils import display_image
from src.domain.shared.pipelines import InferencePipeline
from src.service.dataset_service import image_datasets


# This list must be synchronized with the function create_image_processing_inference_pipelines()
ALL_IMAGE_INFERENCE_PIPELINE_NAMES = [
    "blur_gradients",
    "blur_laplacian",
    "brightness"
]
# Modify this by selecting a subset of desired models from the list above
SELECTED_IMAGE_INFERENCE_PIPELINE_NAMES = [
    "blur_gradients",
    "blur_laplacian"
]


# Logger in global scope to be used by all functions
logger = Logger(name="multimodal_service_logger_inference",
                log_file=LOGS_PATH / "multimodal_service_blur_inference.log")


def create_image_processing_inference_pipelines(pipeline_names: Optional[List[str]] = None) -> List[InferencePipeline]:
    """Factory function to create image processing pipelines."""
    # TODO/FIXME: This factory could be improved with the use of Singletons
    if pipeline_names is None:
        pipeline_names = ALL_IMAGE_INFERENCE_PIPELINE_NAMES

    pipelines = {}

    if "blur_gradients" in pipeline_names:
        blur_inference_pipeline = blur.create_blur_inference_pipeline(
            model_type="gradients",
            estimator_yaml_path=CONFIG_PATH / "blur_estimator_gradients.yaml",
            load=True,
            logger=logger
        )
        pipelines["blur_gradients"] = blur_inference_pipeline
    if "blur_laplacian" in pipeline_names:
        blur_inference_pipeline = blur.create_blur_inference_pipeline(
            model_type="laplacian",
            estimator_yaml_path=CONFIG_PATH / "blur_estimator_laplacian.yaml",
            load=True,
            logger=logger
        )
        pipelines["blur_laplacian"] = blur_inference_pipeline
    if "brightness" in pipeline_names:
        # TODO: We can extend this to include other image processing pipelines
        pass

    return pipelines


# This global dictionary contains all the image processing pipelines
image_inference_pipelines = create_image_processing_inference_pipelines(
    pipeline_names=SELECTED_IMAGE_INFERENCE_PIPELINE_NAMES
)


def predict_image(image: Union[Image.Image, np.ndarray]) -> dict:
    """Process the image by running all image domain pipelines/models and predict its properties.

    Args:
        image: The input image.

    Returns:
        A dictionary containing all the processed image metrics.
    """
    image_metrics = {}

    # Convert to np.ndarray, if necessary
    image_array = image
    if isinstance(image, Image.Image):
        image_array = np.array(image)

    # Loop over all image processing pipelines
    for pipeline_name, pipeline in image_inference_pipelines.items():
        image_metrics[pipeline_name] = pipeline.run_inference_pipeline(image_array).tolist()[0]

    return image_metrics


def predict(image: Union[Image.Image, np.ndarray]) -> dict:
    """Predict image properties. Alias of predict_image()."""
    return predict_image(image)


def run_example() -> None:
    """Run an example of processing an image."""
    blur_dataset = image_datasets.blur
    class_names = blur_dataset.class_names
    logger.info(f"Image class names: {class_names}")

    # Load an image from dataset
    idx = 0
    image, label = blur_dataset[idx]
    display_image(image)

    # Predict
    pred = predict_image(image)
    logger.info(f"Prediction of image with id {idx}: {pred}")
    logger.info(f"Class in model gradients: {class_names[pred['blur_gradients']]}")
    logger.info(f"Class in model laplacian: {class_names[pred['blur_laplacian']]}")

    # Load an image from a file
    image_path = DATA_PATH / pathlib.Path("kaggle_kwentar_blur_dataset/blur_dataset_resized_300/defocused_blurred/0_IPHONE-SE_F.JPG")  # noqa: E501
    image = blur_dataset.load_and_prepare_image(image_path)

    # Predict
    pred = predict_image(image)
    logger.info(f"Prediction of image loaded from path: {pred}")


if __name__ == "__main__":
    run_example()
