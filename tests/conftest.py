"""Configuration file for the pytest tests in the project.

Author: Mikel Sagardia
Date: 2024-12-04
"""
import pathlib
from typing import Any, Dict, List
from PIL import Image
import pytest

import numpy as np
import cv2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.domain.shared.data import (Dataset,
                                    DataTransformer,
                                    ImageDataset,
                                    ImageResizer)
from src.domain.shared.estimators import RuleBasedModel


# Path/file/folder definitions
BASE_TESTS_PATH = pathlib.Path(__file__).resolve().parent
TESTS_DATA_PATH =  BASE_TESTS_PATH / "data"
TESTS_ARTIFACTS_PATH = BASE_TESTS_PATH / "artifacts"
TESTS_CONFIG_PATH = BASE_TESTS_PATH / "config"
BLUR_DATASET_YAML = TESTS_CONFIG_PATH  / "blur_dataset_kaggle.yaml"
BLUR_ESTIMATOR_YAML = TESTS_CONFIG_PATH  / "blur_estimator_gradients.yaml"
BLUR_DATASET_URI = pathlib.Path("blur_dataset_resized_300_subset")
BLUR_DATASET_PATH = TESTS_DATA_PATH / BLUR_DATASET_URI
IMAGE_SAMPLE_BLUR = BLUR_DATASET_PATH / "defocused_blurred" / "0_IPHONE-SE_F.JPG"
IMAGE_SAMPLE_SHARP = BLUR_DATASET_PATH / "sharp" / "0_IPHONE-SE_S.JPG"

DUMMY_FEATURE_NAMES = ["sum"]

# Dummy DataTransformer
class DummyImageTransformer(DataTransformer):
    """Dummy transformer that sums the pixel values of an image."""

    def __init__(self,
                 feature_names: List[str] = DUMMY_FEATURE_NAMES,
                 scale: float = 1.0):
        """Initialize the transformer with the feature names and scale."""
        super().__init__(feature_names=feature_names)
        self._scale = scale

    def transform(self,
                  X: Any,
                  y: Any = None) -> np.ndarray:
        """Transform the input data into the desired features."""
        image_features = None
        if isinstance(X, List):
            image_features = [np.sum(np.array(image))*self._scale for image in X]
        else:
            image_features = np.sum(np.array(X))*self._scale
            image_features = np.expand_dims(image_features, axis=0)

        return np.array(image_features)

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the transformer."""
        return {"scale": self._scale}

    def set_params(self, **params: Any) -> None:
        """Set the parameters of the transformer."""
        if "scale" in params:
            self._scale = params["scale"]


# Dummy RuleBasedModel
class DummyRuleBasedModel(RuleBasedModel):
    """Dummy rule-based model that predicts based on a threshold."""

    def __init__(self, threshold: float = 0.5):
        """Initialize the model with the threshold."""
        self._threshold = threshold
        super().__init__()

    def set_params(self, **params: Any) -> None:
        """Set the parameters of the model."""
        if "threshold" in params:
            self._threshold = params["threshold"]

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the model."""
        return {"threshold": self._threshold}

    def predict(self, X: Any) -> np.ndarray:
        """Predict the output based on the input data."""
        # Example rule-based prediction logic: basic inequality/threshold check
        pred = None
        if isinstance(X, List):
            pred = [1 if x > self._threshold else 0 for x in X]
        else:
            pred = int(self._threshold < X)
            pred = np.expand_dims(pred, axis=0)

        return np.array(pred)


@pytest.fixture
def data_path() -> pathlib.Path:
    return TESTS_DATA_PATH


@pytest.fixture
def artifacts_path() -> pathlib.Path:
    return TESTS_ARTIFACTS_PATH


@pytest.fixture
def config_path() -> pathlib.Path:
    return TESTS_CONFIG_PATH


@pytest.fixture
def dataset_yaml_path() -> pathlib.Path:
    return BLUR_DATASET_YAML


@pytest.fixture
def dataset_path() -> pathlib.Path:
    return BLUR_DATASET_PATH


@pytest.fixture
def dataset_uri() -> pathlib.Path:
    return BLUR_DATASET_URI


@pytest.fixture
def sample_dataset(dataset_uri) -> Dataset:
    return Dataset(uri=dataset_uri, data_path=TESTS_DATA_PATH)


@pytest.fixture
def sample_image_dataset(dataset_uri) -> Dataset:
    return ImageDataset(uri=dataset_uri, data_path=TESTS_DATA_PATH)


@pytest.fixture
def sample_transformers() -> list:
    return [ImageResizer(min_size=50, keep_aspect_ratio=True)]


@pytest.fixture
def image_blur_path() -> pathlib.Path:
    return IMAGE_SAMPLE_BLUR


@pytest.fixture
def image_blur_numpy(image_blur_path) -> np.ndarray:
    image = cv2.imread(image_blur_path) # BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@pytest.fixture
def image_blur_pil(image_blur_path) -> Image.Image:
    return Image.open(image_blur_path)


@pytest.fixture
def image_sharp_path() -> pathlib.Path:
    return IMAGE_SAMPLE_SHARP


@pytest.fixture
def image_sharp_numpy(image_sharp_path) -> np.ndarray:
    image = cv2.imread(image_sharp_path) # BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@pytest.fixture
def image_sharp_pil(image_sharp_path) -> Image.Image:
    return Image.open(image_sharp_path)


@pytest.fixture
def image_empty_path(tmp_path: pathlib.Path = TESTS_ARTIFACTS_PATH) -> pathlib.Path:
    img_path = tmp_path / "test_image.jpg"
    image = np.ones((100, 100, 3), dtype=np.uint8) * 0
    cv2.imwrite(str(img_path), image)
    return img_path


@pytest.fixture
def image_empty_numpy(image_empty_path) -> np.ndarray:
    image = cv2.imread(image_empty_path) # BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@pytest.fixture
def image_invalid_path() -> pathlib.Path:
    return pathlib.Path("/invalid/path/to/image.jpg")


@pytest.fixture
def dummy_image_transformer() -> DummyImageTransformer:
    return DummyImageTransformer(scale=1.0)


@pytest.fixture
def dummy_rule_based_model() -> DummyRuleBasedModel:
    return DummyRuleBasedModel(threshold=0.5)


@pytest.fixture
def dummy_sklearn_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ])
