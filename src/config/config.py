import os
import pathlib
import yaml
from typing import Any, Dict, Union, List, Optional
from pydantic import BaseModel, HttpUrl

from ..core import (
    CONFIG_PATH
)


def load_config_yaml(filename: Union[str, pathlib.Path]) -> dict:
    """Load a YAML configuration file."""
    filename = pathlib.Path(filename) if isinstance(filename, str) else filename
    filepath = CONFIG_PATH / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Config file {filepath} does not exist.")
    try:
        with open(filepath, "r") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {filepath}: {e}") from None
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading the config file {filepath}: {e}") from None


class SuggestParams(BaseModel):  # noqa: D101
    suggest: str
    low: Union[int, float]
    high: Union[int, float]
    log: bool = False


# This is the hp configuration for the XGBoost model
# It is not forced currently
class XGBParams(BaseModel):  # noqa: D101
    objective: str
    eval_metric: str
    n_estimators: SuggestParams
    max_depth: SuggestParams
    learning_rate: SuggestParams
    subsample: SuggestParams
    colsample_bytree: SuggestParams
    gamma: SuggestParams
    reg_alpha: SuggestParams
    reg_lambda: SuggestParams


class Hyperparameters(BaseModel):  # noqa: D101
    nested: bool
    values: Dict[str, Any]


class EstimatorConfig(BaseModel):  # noqa: D101
    modelpath: Optional[str] = None
    transformerspath: Optional[str] = None
    metric: Optional[str] = None
    hyperparameters: Optional[Hyperparameters] = None


class DataTransformerConfig(BaseModel):  # noqa: D101
    params: Dict[str, Any]


class AnnotationProjectConfig(BaseModel):  # noqa: D101
    base_url: Optional[HttpUrl] = "http://localhost:8080"
    api_token: Optional[str] = os.getenv("LABEL_STUDIO_API_TOKEN")
    verify: Optional[bool] = False
    timeout: Optional[int] = 60
    download_samples: Optional[bool] = True
    # sample_identifier is the key in the task data JSON that identifies the sample type
    # In LabelStudio: <Image name="image" value="$image"/> \ <Choices name="choice" toName="image">
    sample_identifier: Optional[str] = "image"
    project_id: int
    sample_storage_uri: str


class DatasetConfig(BaseModel):  # noqa: D101
    uri: str
    class_names: Optional[List[str]] = None
    data_label_mapping: Optional[Dict[str, int]] = None
    labels: Optional[List[Union[str, int]]] = None
    annotation_project: Optional[AnnotationProjectConfig] = None
