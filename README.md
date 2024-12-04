# Multimodal and Multi-Model Machine Learning Service Blueprint

- [Multimodal and Multi-Model Machine Learning Service Blueprint](#multimodal-and-multi-model-machine-learning-service-blueprint)
  - [Introduction](#introduction)
  - [How to Use the Package](#how-to-use-the-package)
    - [Setup](#setup)
    - [Running the API](#running-the-api)
    - [Other Usage Examples](#other-usage-examples)
  - [Package Structure](#package-structure)
    - [Notes and Conventions](#notes-and-conventions)
    - [Cloud Architecture](#cloud-architecture)
  - [Testing and Linting](#testing-and-linting)
    - [Nox](#nox)
    - [Pytest](#pytest)
    - [Test Coverage](#test-coverage)
  - [Improvements and TO-DOs](#improvements-and-to-dos)
  - [Interesting Links](#interesting-links)
  - [License and Authorship](#license-and-authorship)

## Introduction

TBD.

## How to Use the Package

### Setup

```bash
# On Mac, you'll need to explicitly install libomp
# Additionally, make sure that your Anaconda/M iniforge is for the right architecture (i.e., ARM64 for M1+)
brew install libomp

# Create environment (Python 3.11, pip & pip-tools)
conda env create -f conda.yaml
# Activate environment
conda activate multimodal

# Generate pinned dependencies and install/sync
pip-compile requirements-dev.in --verbose
pip-sync requirements-dev.txt

# Alternatively:
pip install -r requirements-dev.txt

# Install package as editable: changes are immediately reflected without reinstalling 
pip install -e .

# If we need a new dependency,
# add it to requirements.in or requirements-dev.in 
# (WATCH OUT: try to follow alphabetical order)
# And then:
pip-compile requirements-dev.in
pip-sync requirements-dev.txt
```

### Running the API

:construction:

```bash
cd .../multimodal_ml_service/
./start_mlflow_tracker.sh

cd .../multimodal_ml_service/
./start_image_api.sh
# Open and try the notebook notebooks/try_api.ipynb



# Example with blur detection PoC
cd .../multimodal_ml_service/
python src/domain/image/blur.py

# Alternatively
python -m src.domain.image.blur

python src/service/image_training_service.py

python src/service/image_inference_service.py

```

```python
API_URL = "http://localhost:8000"
image_path = "path/to/image.jpg"

base64_image = base64.b64encode(image_file.read()).decode("utf-8")
payload = { "image": base64_image}

response = requests.post(f"{API_URL}/predict_image/", json=payload)
results = response.json()

for pipeline_name, result in response.json():
    print(f"{pipeline_name}: {result}")
```


### Other Usage Examples


## Package Structure

Domain Driven Design Structure:

- Adapters: Provide interfaces to interact with external systems, making the core logic independent of external APIs.
- Domain: Core logic specific to the problem domain. Each subdomain (image, 3D models, etc.) is isolated. There is a common `shared` subdomain which builds all the components necessary for ETL, training, evaluation, and inference.
- Service: Handles orchestration and coordination of tasks, such as running ML models or rule-based assessments.
- Entry-points: Handles requests from the external environment (Flask API, command line, etc.) to trigger services.
- Config: Centralized management for different environment settings.
- Tests: Structured testing to ensure each component works independently and as part of the integrated system.
- Scripts: Entry scripts to perform manual operations and setup tasks.

```
repository/
    src/                               # Source folder for package
        core.py                        # General definitions/constants (e.g., paths)
        adapters/                      # Interface for interacting with external systems
            data_repo.py               # Repository for managing images
            db.py                      # Database adapter for general CRUD operations
            annotations.py             # Label-Studio
            serialization.py           # De/Serialization utilities
            logger.py                  # Loggers
            tracker.py                 # ModelTracker based on MLflow
        domain/                        # Core business logic
            image/                     # Subdomain for predicting/processing image properties
                blur.py                # Check image blur quality
                brightness.py          # Assess brightness quality of images
                ...
            3d_models/                 # Subdomain for predicting/processing 3D model properties
                ...
            shared/                    # Shared logic used across domains
                data.py                # Dataset classes
                estimators.py          # Core model classes for assessments
                training.py            # Trainer and TrainingArguments classes
                pipelines.py           # Training and inference pipeline classes
                evaluation.py          # Metrics and Evaluator class
                image_utils.py         # Resizing, etc.
        service/                           # Application services that handle orchestration
            training_service.py            # Orchestrate model training processes
            evaluation_service.py          # Orchestrate evaluation processes
            quality_assessment_service.py  # Handle different quality assessment pipelines (image, 3D scan, build)
        entrypoints/                       # External access points to the system (Flask, CLI, etc.)
            api.py                         # FastAPI app for serving models + endpoints
            cli.py                         # CLI, Click
            gui.py                         # Streamlit
        config/                                 # Configuration files for managing different environments and parameters
            config.py                           # General configuration parser
            blur_dataset.yaml                   # Config for blur dataset
            blur_estimator_gradients.yaml       # Config for blur estimator based on gradients
            ...
    assets/                                 # Images, etc.
    data/                                   # Local datasets
    scripts/                                # Scripts for various utilities and standalone executions
        start_image_api.sh                  # Start FastAPI application with endpoints
        start_mlflow_tracker.sh             # Start MLflow server for tracking
    notebooks/                              # Jupyter notebooks for experimentation and exploration
        try_api.ipynb                       # Test notebook where packages functionalities are shown, e.g., API interaction
    artifacts/                              # Local binaries & Co. of models and transformers
    logs/                                   # Logging outputs
    tests/                                  # Unit tests using pytest
        data/                               # Test data for different domains
        domain/                             # Tests for core domain logic
            shared/
            ...
```

![Package Structure](./assets/package_structure.png)

### Notes and Conventions

- `DataTransformer` can be used either inside an Estimator or before it; it has learnable parameters (via `fit()`), but:
  - Prefer inserting the preprocessing functionality into an `Estimator`/Model if we want to learn the parameters (`LaplacianBlurModel`).
  - Prefer using it outside of the `Estimator` if we don't need to learn the parameters (e.g., `GradientExtractor`) and insert it in the `transformers` list.
- Even though a `DataTransformer` can be serialized, when used in the preprocessing or ETL phase, try to save/load it as a `YAML` file.
- Tracking (via `ModelTracker`, based on `mlflow`) happens at the `Trainer` level, not at higher levels.
- If the MLflow server is started locally and the 

### Cloud Architecture

The following figure depicts a possible (not tested) cloud deployment architecture.

![AWS Architecture](./assets/cloud_architecture.png)

## Testing and Linting

### Nox

We can use [nox](https://nox.thea.codes/en/stable/) to run different testing and validation tasks usual in CI/CD pipelines.

There is a [`noxfile.py`](./noxfile.py) with a variable `LOCATIONS` in it:

```python
LOCATIONS = "src", "tests", "scripts", "noxfile.py"
```

Here's a brief recipe to use it:

```bash
# Go to package repository
cd .../multimodal_ml_service

# Run all sessions in LOCATIONS
nox

# Run install, ruff and pytest sessions on LOCATIONS
nox -s install ruff pytest

# Run ruff on src/domain
nox -s ruff -- src/domain

# Run ruff and fix code in LOCATIONS
nox -s ruff_fix

# Run pylint on LOCATIONS
nox -s pylint
```

### Pytest

If desired, we can separately run [pytest](https://docs.pytest.org/en/stable/) as follows:

```bash
cd .../multimodal_ml_service
pytest
```

Some notes:

- Pytest configuration: [`pytest.ini`](../pytest.ini)
- Pytest fixtures and general constants/assets used across all tests are defined in [`tests/conftest.py`](../tests/conftest.py)
- A small dataset of images and necessary assets is located in [`tests/data`](../tests/data)
- Temporary (removable) artifacts (models, etc.) output during the tests go to [`tests/artifacts`](../tests/artifacts/)
- All configuration files required for the tests (e.g., dataset metadata, etc.) are located in [`tests/config`](../tests/config/)
- Unit tests from `domain`: [`tests/domain`](../tests/domain)

### Test Coverage

Currently, not all modules of the package have tests. Here's a summary of the coverage:

```
Tested in src.domain.shared
- data:
  - [x] Dataset
  - [x] ImageDataset
  - [-] DataTransformer
  - [x] ImageResizer
  - [x] ImageFormatter
  - [x] (DummyImageTransformer)
- estimators:
  - [-] RuleBasedModel
  - [-] Estimator
  - [x] SklearnPipeEstimator
  - [x] RuleBasedEstimator
  - [x] (DummyRuleBasedModel)
- evaluation:
  - [x] Evaluator
  - [x] get_optimization_direction
- training:
  - [ ] TrainingArguments
  - [ ] Trainer
- pipeline:
  - [ ] TrainingPipeline
  - [ ] InferencePipeline
- image_utils:
  - [x] load_image
  - [x] numpy2pil
  - [x] pil2numpy
  - [x] change_image_channels
  - [x] resize_image
  - [ ] base64_to_image
  - [ ] image_to_base64
  - [ ] resize_images_in_folder
- logger:
  - [ ] Logger
- tracker: 
  - [ ] ModelTracker

Tested in src.domain.image:
- blur
  - [ ] extract_gradients_from_image
  - [ ] extract_laplacian_from_image
  - [ ] LaplacianBlurModel
  - [ ] GradientExtractor
  - [ ] create_blur_estimator
  - [ ] create_blur_training_pipeline
  - [ ] create_blur_inference_pipeline

```

## Improvements and TO-DOs

- [x] Implement `nox` to automate testing and linting.
- [ ] Increase test coverage.
- [ ] The `ModelTracker` uses `mlflow.log_artifact()`, not `mlflow.log_model()`, which disables the usage of the MLflow model registry. Change that to use `mlflow.log_model()`.
- [ ] Dependencies: not all are needed?
- [ ] Environment: use `poerty` instead of `pip-tools`?

## Interesting Links

Maybe, you are interested in some related blueprints and guides of mine:

- []()
- []()

## License and Authorship

Refer to [`LICENSE](./LICENSE).
