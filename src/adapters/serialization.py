"""Serialization and deserialization utilities.

Author: Mikel Sagardia
Date: 2024-12-04
"""
from typing import Any, Union
import pathlib
import joblib
import pickle
import json


def serialize(obj: Any, path: Union[str, pathlib.Path]) -> None:
    """Serialize and persist an object to a file.

    The file extension determines the format
    and the object must be compatible with it.

    Supported serialization formats:
        joblib, pickle (& pkl), json, onnx, torch, keras (& h5)

    Args:
        obj (Any): Object to serialize.
        path (str): Path to save the serialized object.
    """
    if isinstance(path, pathlib.Path):
        path = str(path)

    if path.endswith(".joblib"):
        joblib.dump(obj, path)
    elif path.endswith((".pkl", ".pickle")):
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    elif path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(obj, f)
    elif path.endswith(".onnx"):
        pass  #TODO: ONNX serialization
    elif path.endswith(".torch"):
        pass  #TODO: PyTorch serialization
    elif path.endswith((".h5", ".keras")):
        pass  #TODO: Keras serialization
    else:
        raise ValueError(f"Unsupported serialization format: {path}") from None


def deserialize(path: str) -> Any:
    """Deserialize an object from a file.

    The file extension determines the format
    and the object must be compatible with it.

    Supported de/serialization formats:
        joblib, pickle (& pkl), json, onnx, torch, keras (& h5)

    Args:
        path (str): Path to the serialized object.

    Returns:
        Any: Deserialized object.
    """
    if isinstance(path, pathlib.Path):
        path = str(path)

    if path.endswith(".joblib"):
        return joblib.load(path)
    elif path.endswith((".pkl", ".pickle")):
        with open(path, "rb") as f:
            return pickle.load(f)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    elif path.endswith(".onnx"):
        pass  #TODO: ONNX serialization
    elif path.endswith(".torch"):
        pass  #TODO: PyTorch serialization
    elif path.endswith((".h5", ".keras")):
        pass  #TODO: Keras serialization
    else:
        raise ValueError(f"Unsupported serialization format: {path}") from None
