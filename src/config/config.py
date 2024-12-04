import pathlib
import yaml
from typing import Any, Dict, Union, List, Optional
from pydantic import BaseModel

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


class DatasetConfig(BaseModel):  # noqa: D101
    uri: str
    class_names: Optional[List[str]] = None
    folder_labels: Optional[Dict[str, int]] = None
    labels: Optional[List[Union[str, int]]] = None
