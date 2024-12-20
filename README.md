# Multimodal and Multi-Model Machine Learning Service Blueprint

- [Multimodal and Multi-Model Machine Learning Service Blueprint](#multimodal-and-multi-model-machine-learning-service-blueprint)
  - [Introduction](#introduction)
  - [How to Use the Package](#how-to-use-the-package)
    - [Setup](#setup)
    - [Environment Variables](#environment-variables)
    - [Running the Services and the API](#running-the-services-and-the-api)
  - [Package Structure](#package-structure)
    - [Machine Learning Domain Components](#machine-learning-domain-components)
    - [Adapters](#adapters)
    - [Domain Levels and Guidelines to Extend the Package](#domain-levels-and-guidelines-to-extend-the-package)
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

:warning: **Disclaimer: This repository is not about multimodal vision-language models, but about an architecture which enables running multiple models (each of the same of different modalities).**

Imagine a construction company that would like to document and validate onsite construction works automatically: skilled workers build different parts of the buildings and they capture images, videos, 3d scans, or even audio clips (e.g., for reverberation assessments) at different stages of the process. We could build an automatic validation service that evaluates the properties of the captured building works; technically, one approach could be to design that service as an API which takes in several data types (or modalities) and predicts their features by using several models.

This repository presents the architecture of a service which addresses situations similar to the introduced and provides with a blueprint implementation:

- The service can handle several modality inputs (image, 3D models, etc.).
- It can run several models in the background which predict/obtain properties of the input data.
- The models can be typical machine-learning-based models (i.e., neural networks, tree-based models, etc.) or rule-based (i.e., metrics are obtained and algebraically expressed rules used to derive properties).
- All model parameters can be trained, persisted, and used later for inference in an API.

The [Domain-Driven Design (DDD)](https://en.wikipedia.org/wiki/Domain-driven_design) architecture patterns are used, adapted to the usual pipelines and lifecycle required by machine learning (ML) projects:

- DDD separates the code in layers: in the core, we have the *domain* or business-case-related code, which is abstracted to build *services*. Additionally, we have interfaces which interact with the domain components, such as *adapters* that connect to external services, or *entrypoints* which expose our services to the users.
- Machine Learning is characterized by two main operation modes: *training* and *inference*. In the first, dataset samples are *preprocessed* or *transformed*, and fed into an *estimator*, which is expected to predict the same output as the ground truth label; if not, the error is used to tune the *parameters or weights* of the underlying *model*. The product of the training process are precisely those *model parameters*. In the second operation mode, those optimized parameters are loaded to the *estimator*; then, a new *transformed* sample fed to it should yield a correct property prediction &mdash; hopefully :wink:. Both operation modes share many components; these components can be generalized for many domain uses cases in which only the data inputs and the transformer and estimator specifications are changed. That's precisely how the ML pipelines are integrated into the DDD architecture in this blueprint. 

In addition, by leveraging principles and techniques from the [Object-Oriented Programming](https://en.wikipedia.org/wiki/Object-oriented_programming) paradigm, we have a clean separation of the different modules, allowing for easier extension and maintainability.

![Domain-Driven Design and Machine Learning](./assets/ddd_ml.png)

More details on the architecture are provided in [Package Structure](#package-structure). In the next section, the I show how to set up the environment and run the blueprint examples.

:warning: Some final caveats:

- This is a basic template, i.e., don't expect a finished application running on the cloud. Even though some guidelines in that respect are outlined in [Cloud Architecture](#cloud-architecture), the present example ends with a locally running FastAPI service.
- Machine Learning (ML) methods are not in the focus of this blueprint, i.e., no fancy ML models are applied; instead, the key contribution is the architecture and the generalized ML modules than can be reused in many applications. Exemplarily, a simple blur detection method is implemented by extracting Sobel and Laplace features.

## How to Use the Package

In the following, these sections are presented:

- [Setup](#setup) shows how to install the required Python environment and its dependencies.
- [Running the API](#running-the-api) shows how to start using the package by interacting with the FastAPI application.

### Setup

We need to create a dedicated Python environment; here's a quick recipe using [conda](https://docs.conda.io/en/latest/) and [pip-tools](https://github.com/jazzband/pip-tools):

```bash
# On Mac, you'll need to explicitly install libomp
# Additionally, make sure that your Anaconda/Miniforge is for the right architecture (i.e., ARM64 for M1+)
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

### Environment Variables

A `.env` file is expected in the root directory with the following variables:

```bash
# FastAPI
API_HOST="0.0.0.0"
API_PORT="8000"
# MLflow Tracking
MLFLOW_HOST="0.0.0.0"
MLFLOW_PORT="5001"
# ObjectStorage: AWS S3
AWS_ACCESS_KEY_ID="your_access_key_id"
AWS_SECRET_ACCESS_KEY="your_secret_access_key"
AWS_SESSION_TOKEN="your_session_token"
AWS_DEFAULT_REGION="us-east-1"
# AnnotationProject: Label Studio
LABEL_STUDIO_API_TOKEN="your_label_studio_api_token"
```

### Running the Services and the API

First, we need to start the [MLflow](https://mlflow.org/docs/latest/index.html) tracking server:

```bash
# Go to repository folder and activate the environment
cd .../multimodal_ml_service/
conda activate multimodal

# Start the MLflow server
./start_mlflow_tracker.sh
```

Then, in a new Terminal, we can try the *training* and *inference* services as follows:

```bash
# Go to repository folder and activate the environment
cd .../multimodal_ml_service/
conda activate multimodal

# Run Training
python src/service/image_training_service.py

# Run Inference
python src/service/image_inference_service.py
```

Currently, only a simple domain use case is implemented: **blur** detection, i.e., we feed an image and the model predicts whether it's blurry or not.

In addition to using the service modules, we can start the FastAPI server and feed an image to the `/predict_image` endpoint to know about its *blurryness*. To that end, first we start the API from a new Terminal:

```bash
# Go to repository folder and activate the environment
cd .../multimodal_ml_service/
conda activate multimodal

# Start FastAPI server
./start_image_api.sh
```

Then, we can try the API by running the notebook [`try_api.ipynb`](./notebooks/try_api.ipynb), which contains a code similar to the following:

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

In some other modules such as [`blur.py`](./src/domain/image/blur.py) or [`tracker.py`](./src/adapters/tracker.py) there are some `run_example()` functions which showcase some additional functionalities. These can be run by executing the corresponding modules, e.g.:

```bash
# Go to repository folder and activate the environment
cd .../multimodal_ml_service/
conda activate multimodal

# Run Training
python src/domain/image/blur.py
```

## Package Structure

The package is structured in the following subfolders, following the DDD paradigm:

- **Adapters**: They provide interfaces to interact with external systems, making the core logic independent of external APIs. Here's where the logger and the tracker are located, as well as the abstractions/connections to databases (unimplemented) or the like.
- **Domain**: Core logic specific to the domain problems; currently only image blur detection is implemented as example. Each subdomain (image, 3D models, etc.) is isolated. There is a common `shared` subdomain which builds all the **Machine Learning** components necessary for ETL, training, evaluation, and inference. These are explained in more detail in [Machine Learning Domain Components](#machine-learning-domain-components).
- **Service**: It handles orchestration and coordination of tasks, such as running the training or inference pipelines of the desired domain cases.
- **Entry-points**: It handles requests from the external environment (Flask API, command line, etc.) which trigger services.
- Config: Centralized management for different environment settings.
- Tests: Structured testing to ensure each component works independently and as part of the integrated system.
- Scripts: Entry scripts to perform manual operations and setup tasks.

```
repository/
    src/                               # Source folder for package
        core.py                        # General definitions/constants (e.g., paths)
        adapters/                      # Interface for interacting with external systems
            data_repo.py               # Repository for managing images (unimplemented)
            db.py                      # Database adapter for general CRUD operations (unimplemented)
            annotations.py             # Label-Studio (unimplemented)
            serialization.py           # De/Serialization utilities
            logger.py                  # Loggers
            tracker.py                 # ModelTracker based on MLflow
        domain/                        # Core business logic
            image/                     # Subdomain for predicting/processing image properties
                blur.py                # Check image blur quality
                brightness.py          # Assess brightness quality of images (unimplemented)
                ...
            3d_models/                 # Subdomain for predicting/processing 3D model properties (unimplemented)
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
            cli.py                         # CLI, Click (unimplemented)
            gui.py                         # Streamlit app (unimplemented)
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

### Machine Learning Domain Components

A key contribution of the blueprint is the set of machine learning modules contained in `src/domain/shared`.
All the domain modules that solve business cases, such as [`blur.py`](./src/domain/image/blur.py) (which determines whether an image is blurred or not), use these generic machine learning components.

In the following, a brief description of the classes contained in the machine learning modules is provided:

- [`data.py`](./src/domain/shared/data.py)
  - `DataTransfomer`: This abstract class is the template for other transformer classes which perform modifications on the data; an example derived class is `ImageResizer`, which resizes an image. The `DataTransformer` is derived from the `sklearn` classes `BaseEstimator` and `TransformerMixin`, so it requires the definition of `fit()` and `transform()` methods, and it can be used in a `sklearn.Pipeline`.
  - `Dataset` and `ImageDataset(Dataset)`: These classes build and provide sample-label pairs; `Dataset` is for sample filenames and `ImageDataset` for images. The latter can either use a lazy image loader or load the entire dataset, if possible (due to memory limitations).
- [`estimators.py`](./src/domain/shared/estimators.py)
  - `Estimator`: This is the abstract or base class for all models that can be `fit()` to the data and `predict()` properties of new samples. All derived classes contain a model, which can belong to any other framework (`sklearn`, `torch`, etc.). Additionally, rule-based models can be supported, too.
  - `SklearnPipeEstimator(Estimator)`: This is a concrete `Estimator` which contains and operates on a `sklearn.Pipeline` under the hood. We can similarly implement other estimators: `PytorchEstimator`, `ONNXEstimator`, etc.
  - `RuleBasedModel` and `RuleBasedEstimator(Estimator)`: `RuleBasedEstimator` is an `Estimator` which contains and operates with a `RuleBasedModel`. This `RuleBasedModel` is an abstract class which should be derived to define models that are formed by closed-form algebraic conditions. These conditions should be defined by few parameters and can be as simple as inequalities based on threshold values. The key idea here is that these rule-based models are treated like traditional machine learning models; even though they cannot be trained as such (so their `fit()` method returns the model itself), their parameters can be optimized with hyperparameter tuning.
- [`training.py`](./src/domain/shared/training.py)
  - `TrainingArguments`: This class is a container of the hyperparameters for the `Estimator`. Since hyperparameter tuning is supported, the class is able to handle hyperparameter spaces as well as concrete/fixed hyperperameter values.
  - `Trainer`: This class is responsible of running the hyperparameter optimization (with `optuna`) and the training of any `Estimator`. The `ModelTracker` is used by the `Trainer` to log all the metrics and artifacts (by using `mlflow`).
- [`pipelines.py`](./src/domain/shared/pipelines.py)
  - `TrainingPipeline`: It takes all objects necessary to train an `Estimator` and runs the actual training with or without hyperparameter optimization. Then, the produced artifacts (the trained model and transformers) are persisted.
  - `InferencePipeline`: It takes the output artifacts from the `TrainingPipeline` and produces a pipeline object that, given a new input sample, is able to output its prediction.

A detailed relationship of the classes is shown in the following diagram:

![Package Structure](./assets/package_structure.png)

### Adapters

As mentioned, adapters are interfaces to external services. Currently, these classes are defined:

- [`tracker.py`](./src/adapters/tracker.py): `ModelTracker`. This class is a `mlflow` wrapper which logs metrics, parameters and any kind of artifacts.
- [`logger.py`](./src/adapters/logger.py): `Logger`. This class is a wrapper of the standard Python `logging`; it can be easily modified to use any other loggers under the hood.
- [`storage.py`](./src/adapters/storage.py): `ObjectStorage`. This module contains several useful functions to interact with AWS S3. Specifically, `ObjectStorage` represents an S3 bucket.
- [`annotations.py`](./src/adapters/annotations.py): `AnnotationProject`: This module contains several useful functions to interact with a Label Studio server instance, started as an external service. Specifically, `AnnotationProject` represents a Label Studio project. 
- [`data_repo.py`](./src/adapters/data_repo.py): `DataReposirtory`. This class uses `ObjectStorage` and `AnnotationProject` to represent a remote distributes dataset, which can be downloaded and passed to `Dataset`, so that it is accessible by any other object/component in the package.

### Domain Levels and Guidelines to Extend the Package

At the end of the day, the domain is the core part where the business cases are solved.
Once the supporting infrastructure is defined, the goal is to extend the domain codebase to solve the business cases.
We have three hierarchical *domain levels* in the package:

1. **Subdomain**, e.g., the subfolder `image/`, which contains use case modules related to image properties.
2. **Use case**, e.g., the module `blur.py`, which contains models/estimators that detect blur in images. In other words, we are trying to address a business case/problem.
3. **Model/Estimator**, e.g. the blur predictors `LaplacianBlurModel` or the `SklearnPipeEstimator` which takes the features generated by the `GradientExtractor`. In other words, we implement different methods to solve a business problem.

As we can see, only the image subdomain is implemented currently, in which the use case of blur detection is integrated using two models/estimators.

To extend the package focusing on the domain aspects, we can further develop the *domain levels* as follows:

- Model/Estimator level: We can add a new model to the `blur.py` use case module in the `image` subdomain. One example would be implementing a new method for blur detection using the Fourier transform; a possible recipe could be:
  - Create the `FourierExtractor` `DataTransformer` which performs the Fourier transform to an image and selects relevant features. The example to follow is `GradientExtractor`, implemented in `blur.py`.
  - Feed the features to a `SklearnPipeEstimator`, which contains a classifier. Again, a similar example is implemented in `blur.py`.
- Use case level: We can add a new use case in the `image` subdomain, e.g., `brightness.py`, which would detect whether the lighting is correct or not an image. Implementing such a module consists in following the current `blur.py` structure and functions.
- Subdomain level: Add a new subdomain in the same level as `image`, e.g., a new modality folder, such as `3d_model/`. This new folder would contain new modules relative to the new subdomain/modality, which would be implemented following `blur.py`.

We can imagine much more sophisticated domain/business use cases:

- Is the electric wiring correct or not?
- Is a required object present in the 3D scan or not?
- etc.

In any case, the current [`blur.py`](./src/domain/image/blur.py) is the reference example which other cases should follow, based on the three *domain levels*.

:warning: The current [`blur.py`](./src/domain/image/blur.py) uses many factory functions to create the ML objects introduced in the section [Machine Learning Domain Components](#machine-learning-domain-components). One straightforward improvement of the current implementation consists in refactoring and generalizing these factory functions to be used by new business case or domain modules.

### Notes and Conventions

- `DataTransformer` can be used either inside an Estimator or before it; it has learnable parameters (via `fit()`), but:
  - Prefer inserting the preprocessing functionality into an `Estimator`/Model if we want to learn the parameters (`LaplacianBlurModel`), e.g., using `SklearnPipeEstimator`.
  - In general, prefer using them outside of the `Estimator` if we don't need to learn the parameters (e.g., `GradientExtractor`) and insert them in the `transformers` list associated to the `Training/InferencePipeline`.
- Even though a `DataTransformer` can be serialized, when used in the preprocessing or ETL phase, try to save/load it as a `YAML` file.
- Tracking (via `ModelTracker`, based on `mlflow`) happens at the `Trainer` level, not at higher levels.

### Cloud Architecture

The following figure depicts a possible (not tested) cloud deployment architecture. It is still an open task to further elaborate the draft and implement the basic infrastructure necessary for the deployment.

![AWS Architecture](./assets/cloud_architecture.png)

## Testing and Linting

[Nox](#nox) is leveraged for automated 

- linting (mainly with [ruff](https://github.com/astral-sh/ruff))
- and testing (with [pytest](https://docs.pytest.org/en/stable/)).

### Nox

[Nox](https://nox.thea.codes/en/stable/) makes possible to run different testing and validation tasks usual in CI/CD pipelines.

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
  - [ ] save_image

Tested in src.adapters
- logger:
  - [ ] Logger
- tracker: 
  - [ ] ModelTracker
- annotations:
  - [ ] AnnotationProject
- storage:
  - [ ] ObjectStorage
- data_repo:
  - [ ] DataRepository

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
- [ ] Implement the cloud infrastructure code.
- [ ] Increase test coverage.
- [ ] Refactor and generalize the factory functions in `blur.py` so that they are re-usable by other domain modules, e.g., `brightness.py` (unimplemented).
- [ ] The `ModelTracker` uses `mlflow.log_artifact()`, not `mlflow.log_model()`, which disables the usage of the MLflow model registry. Change that to use `mlflow.log_model()`.
- [ ] The `Trainer` should receive a `Dataset` as training argument so that we can use lazy loading of batches during training of ANNs.
- [ ] Dependencies: not all are needed in the environment definition?
- [ ] Environment: use `poerty` instead of `pip-tools`?

## Interesting Links

Maybe, you are interested in some related blueprints and guides of mine:

- Design Patterns in Python: [mxagar/design_patterns_notes](https://github.com/mxagar/design_patterns_notes).
- Notes on the Udacity MLOps Nanodegree: [mxagar/mlops_udacity](https://github.com/mxagar/mlops_udacity).

## License and Authorship

Refer to [`LICENSE`](./LICENSE).
