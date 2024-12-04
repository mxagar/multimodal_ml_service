"""Base configuration for the package.

Author: Mikel Sagardia
Date: 2024-12-04
"""
import pathlib

# Path definitons
SRC_PATH = pathlib.Path(__file__).resolve().parent
TRACKING_PATH = pathlib.Path(__file__).resolve().parent.parent / "tracking"
DATA_PATH =  pathlib.Path(__file__).resolve().parent.parent / "data"
ARTIFACTS_PATH = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
TESTS_PATH = pathlib.Path(__file__).resolve().parent.parent / "tests"
LOGS_PATH = pathlib.Path(__file__).resolve().parent.parent / "logs"
CONFIG_PATH = pathlib.Path(__file__).resolve().parent / "config"
ENTRYPOINTS_PATH = pathlib.Path(__file__).resolve().parent / "entrypoints"
SERVICE_PATH = pathlib.Path(__file__).resolve().parent / "service"
DOMAIN_PATH = pathlib.Path(__file__).resolve().parent / "domain"
ADAPTERS_PATH = pathlib.Path(__file__).resolve().parent / "adapters"

VERSION = "0.0.1"
APPLICATION_NAME = "multimodal_ml_service"
