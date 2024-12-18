"""Custom training/model tracker and associated utilities.

Main class: ModelTracker
Entry point factory function: create_model_tracker()

Before using the ModelTracker, we need to start the MLflow server:

    scripts/start_mlflow_server.sh

An example is provided at the end of the file to demonstrate how to use the ModelTracker.

Author: Mikel Sagardia
Date: 2024-12-04
"""
import os
import sys
import subprocess
from typing import Any, Dict, Optional, List
import pathlib
from dotenv import load_dotenv
import requests

import mlflow

from src.core import SRC_PATH
from src.adapters.logger import Logger
from src.adapters.serialization import serialize, deserialize
from src.core import TRACKING_PATH, ARTIFACTS_PATH, LOGS_PATH


# Load environment variables
load_dotenv(override=True, dotenv_path=str(SRC_PATH / ".env"))


class ModelTracker:
    """Model tracker to log model parameters, metrics, and artifacts.

    It uses the MLflow library to log the data to a tracking server.
    However, this version does not use the MLflow model logging methods, i.e., we cannot use the model registry.
    Instead, we use log_artifact() to log the model as a file in the background.

    Some guidelines:
    - log_metric("f1", 0.9): Log numerical metrics (e.g., accuracy, loss) for tracking over steps.
        - Requires local file? No
        - Generates a file? No, it stores the data in the tracking server DB.
    - log_params({"epochs": 10, "lr": 0.1}): Log flat key-value pairs as parameters for the experiment.
        - Requires local file? No
        - Generates a file? No, it stores the data in the tracking server DB.
    - log_dict({"architecture": "resnet18", "classes": ['blurred', 'sharp']}): Log a structured dictionary as artifact.
        - Requires local file? No
        - Generates a file? Yes, it generates a JSON file.
    - log_artifact(local_path="../outputs/image.png", artifact_path="conf_matrix.png"):
      Log an artifact (= a local file/directory to the artifact store under artifacts/ (server).
        - Requires local file? Yes
        - Generates a file? No, the file is copied from local_path to the artifact_path under artifacts/ (server).
    - log_model(): It calls log_artifact() to log the model as a file in the background.
        - Requires local file? In mlflow, it doesn't, but here it does, via log_artifact().
        - Generates a file? No, the file is copied from local_path to the artifact_path under artifacts/ (server).

    Note about log_model() and log_artifact(): MLflow has its own model logging methods, but instead of using them,
    I have used log_artifact() to log the model as a file in the background.
    The reason is that only a given set of model/framework flavors are supported.
    Using this workaround, we control and need to implement custom serialization/deserialization methods,
    but we can log any model native to our application.
    Drawback: We lose the ability to use MLflow's model registry and model flavors.
    TODO, Solution: We should use log_model() and pyfunc/flavor methods.

    For more information on MLflow, you can visit
        https://github.com/mxagar/mlflow_guide

    """

    def __init__(self,
                 tracking_uri: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 logger: Optional[Logger] = None):
        """Initialize the model tracker.

        Args:
            tracking_uri (Optional[str]): URI of the MLflow tracking server.
            experiment_name (Optional[str]): Name of the experiment to use or create.
            logger (Optional[logging.Logger]): Logger instance for logging messages.
        """
        self.tracker = mlflow
        self.logger = logger

        if tracking_uri:
            if self.logger is not None:
                self.logger.info(f"Setting tracking URI to {tracking_uri}")
            self.tracker.set_tracking_uri(tracking_uri)

        if experiment_name:
            if self.logger is not None:
                self.logger.info(f"Setting experiment to {experiment_name}")
            try:
                self.tracker.set_experiment(experiment_name)
            except Exception:
                if self.logger is not None:
                    self.logger.warning(f"Experiment '{experiment_name}' not found, creating a new one.")
                self.tracker.create_experiment(experiment_name)
                self.tracker.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """Start a new MLflow run."""
        try:
            if self.logger is not None:
                self.logger.info(f"Starting run: {run_name}, nested={nested}")
            self.tracker.start_run(run_name=run_name, nested=nested)
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to start run: {e}")

    def end_run(self):
        """End the current MLflow run."""
        try:
            if self.logger is not None:
                self.logger.info("Ending run.")
            self.tracker.end_run()
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to end run: {e}")

    def list_experiments(self) -> Dict[str, Any]:
        """List all experiments in the tracking server."""
        try:
            if self.logger is not None:
                self.logger.info("Listing experiments.")
            return self.tracker.list_experiments()
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to list experiments: {e}")
            return {}

    def list_runs(self, experiment_id: str) -> List:
        """List all runs for a given experiment."""
        try:
            if self.logger is not None:
                self.logger.info(f"Listing runs for experiment: {experiment_id}")
            return self.tracker.search_runs(experiment_ids=[experiment_id])
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to list runs for experiment {experiment_id}: {e}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow.

        The parameters are stored in the DB.
        """
        try:
            if self.logger is not None:
                self.logger.debug(f"Logging parameters: {params}")
            self.tracker.log_params(params)
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to log parameters: {e}")

    def log_dict(self, params: Dict[str, Any], artifact_path: str):
        """Log a dictionary-like structured object to MLflow.

        The params object will be automatically converted to JSON.

        The parameters are stored in the following structure:
            mlruns/ # TRACING_PATH
                <experiment_id>/
                    <run_id>/
                        artifacts/
                            params.json # params
        """
        try:
            if self.logger is not None:
                self.logger.debug(f"Logging dictionary: {params}")
            self.tracker.log_dict(params, artifact_file=artifact_path)
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to log dictionary: {e}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to MLflow.

        The metrics are stored in the DB.
        """
        try:
            if self.logger is not None:
                message = f"Logging metrics: {metrics}"
                if step is not None:
                    message += f" at step {step}"
                self.logger.debug(message)
            self.tracker.log_metrics(metrics, step=step)
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to log metrics: {e}")

    def log_model(self, model: Any, local_path: str | pathlib.Path,
                  artifact_path: Optional[str | pathlib.Path] = None):
        """Log a model to MLflow.

        The model must be serializable as PICKLE, JOBLIB, ONNX, TORCH, or KERAS.
        Also, if the model is dictionary, it can be serialized as JSON.

        MLflow supports logging models in various formats, such as PyTorch, ONNX, and scikit-learn.
        Each model flavor has its own serialization method.
        In order to avoid issues, I have instead used log_artifact() to log the model as a file.
        This can be changed to use the MLflow model logging methods if needed.

        The models are stored in the following structure:
            mlruns/ # TRACING_PATH
                <experiment_id>/
                    <run_id>/
                        artifacts/
                            model.pkl # filename in local_path

        """
        if isinstance(local_path, pathlib.Path):
            local_path = str(local_path)
        if isinstance(artifact_path, pathlib.Path):
            artifact_path = str(artifact_path)

        serialization_format = local_path.split(".")[-1]
        if serialization_format not in ["pkl", "pickle", "joblib", "onnx", "torch", "keras", "h5", "json"]:
            if self.logger is not None:
                self.logger.error(f"Unsupported model serialization format: {serialization_format}")
            raise ValueError(f"Unsupported model serialization format: {serialization_format}") from None

        try:
            if self.logger is not None:
                self.logger.info(f"Logging model to {artifact_path}")
            # Serialize model locally
            serialize(model, local_path)
            # Log model as artifact
            # Alternative, but if model flavor are correctly handled: self.tracker.log_model(model, artifact_path)
            self.log_artifact(local_path=local_path, artifact_path=artifact_path)
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to log model: {e}")

    def log_artifact(self,
                     local_path: str | pathlib.Path,
                     artifact_path: Optional[str | pathlib.Path] = None):
        """Log an artifact to MLflow.

        An artifact is a local file intended to be uploaded to the tracking server.
        Therefore the artifact_path should be a local file path.

        The artifacts are stored in the following structure:
            mlruns/ # TRACING_PATH
                <experiment_id>/
                    <run_id>/
                        artifacts/
                            text.txt # filename in local_path
        """
        if isinstance(local_path, pathlib.Path):
            local_path = str(local_path)
        if isinstance(artifact_path, pathlib.Path):
            artifact_path = str(artifact_path)

        try:
            if self.logger is not None:
                self.logger.info(f"Logging artifact: {local_path}")
            self.tracker.log_artifact(local_path=local_path, artifact_path=artifact_path)
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to log artifact: {e}")

    def get_metrics(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the metrics for a given run."""
        try:
            if self.logger:
                self.logger.info(f"Fetching metrics for run: {run_id}")
            run = mlflow.get_run(run_id)
            return run.data.metrics
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to fetch metrics for run {run_id}: {e}")
            return None

    def get_params(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the parameters for a given run."""
        try:
            if self.logger:
                self.logger.info(f"Fetching parameters for run: {run_id}")
            run = mlflow.get_run(run_id)
            return run.data.params
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to fetch parameters for run {run_id}: {e}")
            return None

    def get_artifact_path(self, run_id: str, artifact_path: str) -> Optional[str]:
        """Get the path to an artifact for a given run."""
        try:
            if self.logger:
                self.logger.info(f"Fetching artifact from {artifact_path} for run: {run_id}")
            local_path = self.tracker.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
            return local_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to fetch artifact path for run {run_id}: {e}")
            return None

    def get_artifact(self, run_id: str, artifact_path: str) -> Optional[Any]:
        """Get the path to an artifact for a given run."""
        local_path = self.get_artifact_path(run_id, artifact_path)
        if local_path is not None:
            return deserialize(local_path)

    def get_dict(self, run_id: str, artifact_path: str) -> Optional[Dict[str, Any]]:
        """Get the dictionary artifact for a given run."""
        return self.get_artifact(run_id, artifact_path)

    def get_model(self, run_id: str, artifact_path: str) -> Optional[Any]:
        """Get the model artifact for a given run."""
        return self.get_artifact(run_id, artifact_path)

    def __enter__(self):
        """Support for context manager (with statement)."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensure the run is ended when exiting the context."""
        self.end_run()


def _get_tracker_env_vars() -> Dict[str, str]:
    """Get the environment variables for the ModelTracker."""
    # Set environment variables
    load_dotenv(override=True, dotenv_path=str(SRC_PATH / ".env"))
    # Read environment variables or set defaults
    # Paths should be absolute
    MLFLOW_BACKEND_URI = "sqlite:///" + str(TRACKING_PATH) + "/mlflow.db"
    MLFLOW_ARTIFACT_ROOT = str(TRACKING_PATH) + "/mlruns"

    # Override with environment variables if set
    MLFLOW_BACKEND_URI = os.getenv("MLFLOW_BACKEND_URI", MLFLOW_BACKEND_URI)
    MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", MLFLOW_ARTIFACT_ROOT)
    MLFLOW_HOST = os.getenv("MLFLOW_HOST", "0.0.0.0")
    MLFLOW_PORT = os.getenv("MLFLOW_PORT", "5001")
    MLFLOW_SERVER_URI = f"http://localhost:{MLFLOW_PORT}" if MLFLOW_HOST == "0.0.0.0" else f"http://{MLFLOW_HOST}:{MLFLOW_PORT}"
    MLFLOW_SERVER_URI = os.getenv("MLFLOW_SERVER_URI", MLFLOW_SERVER_URI)

    return {
        "MLFLOW_HOST": MLFLOW_HOST,
        "MLFLOW_PORT": MLFLOW_PORT,
        "MLFLOW_SERVER_URI": MLFLOW_SERVER_URI,
        "MLFLOW_BACKEND_URI": MLFLOW_BACKEND_URI,
        "MLFLOW_ARTIFACT_ROOT": MLFLOW_ARTIFACT_ROOT,
    }


def _check_mlflow_server(server_uri: str) -> bool:
    """Check if the MLflow server is running at the given URI.

    Args:
        server_uri (str): The URI of the MLflow server, e.g., "http://localhost:5001".

    Returns:
        bool: True if the server is reachable, False otherwise.
    """
    health_endpoint = f"{server_uri}/health"
    try:
        response = requests.get(health_endpoint, timeout=1)
        if response.status_code == 200:  # noqa: PLR2004
            return True
    except Exception:
        return False


def start():
    """Start the MLflow tracking server."""
    mlflow_env = _get_tracker_env_vars()
    MLFLOW_HOST = mlflow_env["MLFLOW_HOST"]
    MLFLOW_PORT = mlflow_env["MLFLOW_PORT"]
    MLFLOW_BACKEND_URI = mlflow_env["MLFLOW_BACKEND_URI"]
    MLFLOW_ARTIFACT_ROOT = mlflow_env["MLFLOW_ARTIFACT_ROOT"]

    # Build the command
    command = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", MLFLOW_BACKEND_URI,
        "--default-artifact-root", MLFLOW_ARTIFACT_ROOT,
        "--host", MLFLOW_HOST,
        "--port", MLFLOW_PORT,
    ]

    # Start the server
    # MLflow server is then running and UI accessible, e.g. at http://localhost:$MLFLOW_PORT
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Running the mlflow server failed: {e}")  # noqa: T201
        raise


def create_model_tracker(experiment_name: str,
                         mlflow_server_uri: Optional[str] = None,
                         logger: Optional[Logger] = None) -> ModelTracker:
    """Create a ModelTracker instance with the given experiment name and logger.

    Args:
        experiment_name (str): Name of the experiment to use or create.
        mlflow_server_uri (Optional[str]): URI of the MLflow tracking server,
            e.g., "http://localhost:5001". If not provided, it should be
            defined in the environment variables as MLFLOW_SERVER_URI.
        logger (Logger): Logger instance for logging messages.

    Returns:
        ModelTracker: The ModelTracker instance.
    """
    # Get environment variables or set defaults
    mlflow_env = _get_tracker_env_vars()
    MLFLOW_SERVER_URI = mlflow_server_uri
    if MLFLOW_SERVER_URI is None:
        MLFLOW_SERVER_URI = mlflow_env["MLFLOW_SERVER_URI"]

    MLFLOW_ARTIFACT_ROOT = mlflow_env["MLFLOW_ARTIFACT_ROOT"]
    MLFLOW_BACKEND_URI = mlflow_env["MLFLOW_BACKEND_URI"]

    if logger is not None:
        logger.info("Connecting to MLflow server...")
        logger.info(f"TRACKING_PATH: {TRACKING_PATH}")
        logger.info(f"LOGGER_PATH: {LOGS_PATH}")
        logger.info(f"ARTIFACTS_PATH: {ARTIFACTS_PATH}")
        logger.info(f"MLFLOW_BACKEND_URI: {MLFLOW_BACKEND_URI}")
        logger.info(f"MLFLOW_ARTIFACT_ROOT: {MLFLOW_ARTIFACT_ROOT}")
        logger.info(f"MLFLOW_SERVER_URI: {MLFLOW_SERVER_URI}")

    # Initialize the ModelTracker
    tracker = ModelTracker(
        tracking_uri=MLFLOW_SERVER_URI,        # MLflow server URI
        experiment_name=experiment_name,       # Experiment name
        logger=logger                          # Pass the custom logger
    )

    return tracker


def run_example() -> None:
    """Run an example of using the ModelTracker.

    This function demonstrates how to use the ModelTracker to log parameters, metrics, models, and artifacts.
    We need to start the MLflow server before running this example, either by calling start()
    or with a command as the following (variables to be modified as needed):

        cd scripts
        ./start_mlflow_server.sh

    The logged data is stored in the following structure (on the server):

        mlruns/ # TRACKING_PATH/mlruns (default: "tracking/mlruns")
            <experiment_id>/
                <run_id>/
                    artifacts/
                        params.json
                        dummy_model.pkl
                        example_artifact.txt

    The data introduced via log_params() and log_metrics() is stored in the MLflow tracking server DB,
    i.e., there is no local file generated.
    The data introduced via log_dict(), log_model(), and log_artifact() generates/requires files.

    After logging them, we can open the MLflow UI by visiting
        http://localhost:5001
    (or the specified host & port) in a web browser.
    """
    # Set up a custom logger
    log_file = LOGS_PATH / "tracker.log"
    logger = Logger(name="mlflow_logger", log_file=log_file).get_logger()

    # Instantiate model tracker
    tracker = create_model_tracker(experiment_name="example_experiment", logger=logger)

    ## -- Tracking

    try:
        # Start a new MLflow run
        # Since we have a context manager, we can use a `with` statement
        # and avoid calling start_run() and end_run() explicitly; see below.
        tracker.start_run(run_name="example_run")

        # Log parameters
        tracker.log_params({
            "learning_rate": 0.01,
            "batch_size": 32,
            "num_epochs": 10
        })

        # Log a dictionary-like object
        tracker.log_dict({
            "model": "CNN",
            "layers": 3,
            "activation": "relu"
        }, artifact_path="params.json")

        # Log metrics over multiple steps (e.g., epochs)
        for epoch in range(1, 11):
            tracker.log_metrics({
                "accuracy": 0.85 + epoch * 0.01,
                "loss": 1.0 - epoch * 0.05
            }, step=epoch)

        # Log a single metric
        tracker.log_metrics({"f1": 0.75})

        # Log a model (as a dummy example)
        dummy_model = {"weights": [0.1, 0.2, 0.3]}  # Replace with a real model
        filename = "dummy_model.joblib"
        local_model_path = ARTIFACTS_PATH / filename
        tracker.log_model(dummy_model,
                          local_path=local_model_path)

        # Log an artifact (e.g., a generated file)
        filename = "example_artifact.txt"
        local_artifact_path = ARTIFACTS_PATH / filename
        with open(str(local_artifact_path), "w") as f:
            f.write("This is an example artifact.")
        tracker.log_artifact(local_path=local_artifact_path)

    finally:
        # Ensure the run is ended
        tracker.end_run()

    ## -- Getting Models and Artifacts

    experiments = tracker.list_experiments()
    for exp in experiments:
        print(f"Experiment ID: {exp.experiment_id}, Name: {exp.name}")  # noqa: T201

    runs_df = tracker.list_runs(experiments[1].experiment_id)
    print(f"Runs: {runs_df.columns}")  # noqa: T201

    try:
        # Specify the run ID (replace with the actual run ID)
        run_id = runs_df.loc[0, "run_id"]

        # Get parameters
        params = tracker.get_params(run_id)
        print(f"Parameters: {params}")  # noqa: T201

        # Get metrics
        metrics = tracker.get_metrics(run_id)
        print(f"Metrics: {metrics}")  # noqa: T201

        # Get dictionary artifact (params.json)
        artifact_dict = tracker.get_dict(run_id, artifact_path="params.json")
        print(f"Artifact Dictionary: {artifact_dict}")  # noqa: T201

        # Get and load a model artifact
        model = tracker.get_model(run_id, artifact_path="dummy_model.joblib")
        print(f"Loaded Model: {model}")  # noqa: T201

        # Get and read an artifact file
        local_artifact_path = tracker.get_artifact_path(run_id, artifact_path="example_artifact.txt")
        with open(local_artifact_path, "r") as f:
            artifact_content = f.read()
        print(f"Artifact Content: {artifact_content}")  # noqa: T201

    except Exception as e:
        raise ValueError(f"Failed to fetch data: {e}") from None

    ## -- Using Context Manager

    mlflow_env = _get_tracker_env_vars()
    MLFLOW_SERVER_URI = None
    if MLFLOW_SERVER_URI is None:
        MLFLOW_SERVER_URI = mlflow_env["MLFLOW_SERVER_URI"]

    with ModelTracker(tracking_uri=MLFLOW_SERVER_URI, experiment_name="example_experiment") as tracker:
        tracker.log_params({
            "learning_rate": 0.01,
            "batch_size": 32,
            "num_epochs": 10
        })


if __name__ == "__main__":
    run_example()
