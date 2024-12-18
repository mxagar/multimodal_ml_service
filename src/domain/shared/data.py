"""Data transformers and dataset classes for handling different types of data sources.

Dataset classes can be used to manage the loading of images, 3D models, as well as any other data type.
Each element in a Dataset class can be a tuple of the data and its label.
Each data type should have its own class derived from the base class Dataset.

Data transformers are Scikit-Learn transformers that can be used preprocess the data, e.g., resizing images.
Even though they are Scikit-Learn transformers, they are not meant to be used only with that framework,
but they can be used with any other component in the project.

Author: Mikel Sagardia
Date: 2024-12-04
"""
from abc import abstractmethod
import pathlib
from typing import Union, List, Optional, Tuple, Any, Dict

import yaml
import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin

from src.core import DATA_PATH
from src.adapters.logger import Logger
from src.config.config import load_config_yaml, DatasetConfig, AnnotationProjectConfig
from src.adapters.data_repo import DataRepository

from src.domain.shared.image_utils import (
    load_image,
    resize_image,
    ChannelOrder,
    ImageFormat,
    numpy2pil,
    pil2numpy,
    change_image_channels
)


class DataTransformer(BaseEstimator, TransformerMixin):
    """Abstract base class for data transformation or feature extraction.

    It can be used for images, 3D models, as well as any other data type.
    """

    def __init__(self, feature_names: Optional[List[str]] = None) -> None:
        super().__init__()
        self._feature_names = feature_names

    # Implement this method, if necessary
    def fit(self, X: Any, y: Any = None) -> Any:  # noqa: D102
        return self

    @abstractmethod
    # Implement this method!
    def transform(self, X: Any, y: Any = None) -> Any:  # noqa: D102
        pass

    @abstractmethod
    # Implement this method!
    def get_params(self) -> dict:  # noqa: D102
        pass

    @abstractmethod
    # Implement this method!
    def set_params(self, **params: Any) -> None:  # noqa: D102
        pass

    def save_params(self, path: Union[str, pathlib.Path]) -> None:  # noqa: D102
        params = self.get_params()
        data = {self.__class__.__name__: {"params": params}}
        with open(path, "w") as file:
            yaml.dump(data, file)

    def load_params(self, path: Union[str, pathlib.Path]) -> None:  # noqa: D102s
        data = load_config_yaml(path)
        class_name = self.__class__.__name__
        if class_name in data:
            params = data[class_name].get("params", {})
            self.set_params(**params)
        else:
            raise ValueError(f"No parameters found for class '{class_name}' in {path}.")

    def set_feature_names(self, feature_names: List[str]) -> None:  # noqa: D102
        self._feature_names = feature_names

    @property
    def feature_names(self) -> Union[None, List[str]]:  # noqa: D102
        return self._feature_names


class ImageResizer(DataTransformer):
    """A DataTransformer implementation that resizes images.

    Usage:

        resizer = ResizeImage(min_size=300, keep_aspect_ratio=True)
        image_resized = resizer.transform(image)
    """

    def __init__(self,
                 min_size: int,
                 keep_aspect_ratio: bool = True) -> None:
        super().__init__()
        self.min_size = min_size
        self.keep_aspect_ratio = keep_aspect_ratio

    def transform(self,
                  X: Union[Union[Image.Image, np.ndarray], List[Union[Image.Image, np.ndarray]]],
                  y: Any = None) -> Union[Union[Image.Image, np.ndarray], List[Union[Image.Image, np.ndarray]]]:
        """Transform the input image(s) to image(s) of the desired size."""
        X_resized = None
        if isinstance(X, List):
            X_resized = [resize_image(image,
                                      min_size=self.min_size,
                                      keep_aspect_ratio=self.keep_aspect_ratio) for image in X]
        else:
            X_resized = resize_image(X,
                                     min_size=self.min_size,
                                     keep_aspect_ratio=self.keep_aspect_ratio)

        return X_resized

    def get_params(self) -> dict:  # noqa: D102
        return {"min_size": self.min_size,
                "keep_aspect_ratio": self.keep_aspect_ratio}

    def set_params(self, **params: Any) -> None:  # noqa: D102
        if "min_size" in params:
            self.min_size = params["min_size"]
        if "keep_aspect_ratio" in params:
            self.keep_aspect_ratio = params["keep_aspect_ratio"]


class ImageFormatter(DataTransformer):
    """A DataTransformer implementation that formats images between NumPy and PIL and adjusts channel orders.

    Usage:

        formatter = ImageFormatter(out_format="PIL", out_channels=ChannelOrder.RGB)
        formatted_image = formatter.transform(image)
    """
    def __init__(self,
                 out_format: str = ImageFormat.PIL, # or "NUMPY"
                 out_channels: ChannelOrder = ChannelOrder.RGB,
                 in_channels: ChannelOrder = ChannelOrder.RGB) -> None:
        super().__init__()
        self.out_format = out_format
        self.out_channels = out_channels
        self.in_channels = in_channels

    def transform(self,
                  X: Union[Union[Image.Image, np.ndarray], List[Union[Image.Image, np.ndarray]]],
                  y: Any = None) -> Union[Union[Image.Image, np.ndarray], List[Union[Image.Image, np.ndarray]]]:
        """Transform the input image(s) to image(s) of the desired format and channel order."""
        X_formatted = None
        if isinstance(X, List):
            X_formatted = [self._convert_image(image) for image in X]
        else:
            X_formatted = self._convert_image(X)

        return X_formatted

    def _convert_image(self, image: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        if isinstance(image, Image.Image):
            if self.out_format == ImageFormat.NUMPY:
                # WARNING
                # If input is PIL, channels should be RGB by default,
                # so no channel conversion is needed
                # FIXME: This is not ideal, maybe we should support any channel order for PIL, too...
                return pil2numpy(image, out_channels=self.out_channels)
        elif isinstance(image, np.ndarray):
            if self.out_format == ImageFormat.PIL:
                return numpy2pil(image, in_channels=self.in_channels)
            elif self.in_channels != self.out_channels:
                return change_image_channels(image,
                                             in_channels=self.in_channels,
                                             out_channels=self.out_channels)
        return image

    def get_params(self) -> dict:  # noqa: D102
        return {"out_format": str(self.out_format.value),
                "out_channels": str(self.out_channels.value),
                "in_channels": str(self.in_channels.value)}

    def set_params(self, **params: Any) -> None:  # noqa: D102
        if "out_format" in params:
            self.out_format = ImageFormat(params["out_format"])
        if "out_channels" in params:
            self.out_channels = ChannelOrder(params["out_channels"])
        if "in_channels" in params:
            self.in_channels = ChannelOrder(params["in_channels"])


def save_data_transformers(transformers: List[DataTransformer],
                           path: Union[str, pathlib.Path],
                           logger: Optional[Logger] = None) -> None:
    """Save a list of transformers to a YAML file. Currently, each type of transformer can appear once in the list."""
    if len(transformers) > 0:
        transformers_data = {}
        for transformer in transformers:
            class_name = transformer.__class__.__name__
            transformers_data[class_name] = {"params": transformer.get_params()}

        with open(path, "w") as file:
            yaml.dump(transformers_data, file)


def load_data_transformers(transformers: List[DataTransformer],
                           path: Union[str, pathlib.Path],
                           debug: bool = False,
                           logger: Optional[Logger] = None) -> None:
    """Load the transformers and set the parameters to them.

    Those params are loaded from a YAML file.
    IMPORTANT: Currently, each type of transformer can appear once in the list.
    """
    if path.exists():
        transformers_data = load_config_yaml(path)

        # Loop over all transformers in the YAML file
        for class_name, data in transformers_data.items():
            # Get the transformer instance from the provided list which matches the instance in the YAML
            matching_transformer = next((t for t in transformers if t.__class__.__name__ == class_name), None)
            # If params in YAML but transformer not passed: Error
            if not matching_transformer:
                raise ValueError(f"Transformer '{class_name}' found in {path} but not in provided transformers list.")
            # Else: Set the parameters
            params = data.get("params", None)
            if params:
                matching_transformer.set_params(**params)

        # If we pass a transformer which has no params in the YAML file: do nothing
        for transformer in transformers:
            if transformer.__class__.__name__ not in transformers_data and debug:
                class_name = transformer.__class__.__name__
                if logger is not None:
                    logger.warning(f"No saved parameters found for '{class_name}' in {path}.")
    elif debug and logger is not None:
        logger.warning(f"No YAML found, returning `transformers` as provided: {path}.")


class Dataset:
    """A base dataset class that handles different types of data sources.

    If the dataset is local, it can be a folder with subfolders, each containing samples.
    If the dataset is remote, we should define it in the YAML file using the `annotation_project` field.
    Then, the data will be downloaded from the cloud using DataRepository.
    NOTE: DataRepository works with Label Studio projects and S3 buckets under the hood,
    which are interfaced in the adapters.

    Usage:
        ds1 = Dataset(uri="path/to/dataset")
        ds2 = Dataset()
        ds2.setup_from_yaml("path/to/dataset.yaml")
        filename, label = ds1[0]  # Get the first item, e.g.: ("path/to/image.jpg", "class_1")
    """

    def __init__(self,
                 uri: Optional[Union[str, pathlib.Path]] = None,
                 labels: Optional[List[str]] = None,
                 data_label_mapping: Optional[Dict] = None,
                 class_names: Optional[List[str]] = None,
                 yaml_path: Optional[Union[str, pathlib.Path]] = None,
                 data_path: Optional[Union[str, pathlib.Path]] = None,
                 load_filenames: bool = True) -> None:

        self._uri = pathlib.Path(uri) if uri is not None else None
        self._labels: str = labels if labels is not None else []
        self._data_label_mapping = data_label_mapping if data_label_mapping is not None else {}
        self._yaml_path = pathlib.Path(yaml_path) if yaml_path is not None else None
        self._data_path = pathlib.Path(data_path) if data_path is not None else DATA_PATH
        self._load_filenames = load_filenames

        self._local_path: pathlib.Path = None
        self._filenames: List[str] = []
        # Class names are the classes used in the dataset
        # folder names are the names of the subfolders in the dataset
        # which does not necessarily correspond to the class names
        self._class_names: List[str] = class_names if class_names is not None else []
        self._folder_names: List[str] = []
        # The Dataset YAML can contain the key annotation_project
        # which connects the dataset to a Label Studio project
        # and downloads the images from the cloud using DataRepository
        self._annotation_project_config: Optional[AnnotationProjectConfig] = None
        # DataRepository is instantiated only if we have an annotation project
        self._data_repository: Optional[DataRepository] = None

        self._current_index = 0

        if self._uri is not None:
            self._setup()
        elif self._yaml_path is not None:
            self.setup_from_yaml(self._yaml_path)

    def setup_from_yaml(self, yaml_path: Union[str, pathlib.Path]) -> None:
        """Setup the dataset from a YAML file."""
        config_dict = load_config_yaml(str(yaml_path))
        config = DatasetConfig(**config_dict) # Validation via Pydantic
        self._uri = config.uri # config_dict.get('uri', None)
        self._data_label_mapping = config.data_label_mapping if config.data_label_mapping is not None else {}
        self._class_names = config.class_names if config.class_names is not None else []
        self._labels = config.labels if config.labels is not None else []
        self._annotation_project_config = config.annotation_project  # None by default, if not set

        if self._uri is None:
            raise ValueError("YAML file must contain a 'uri' field.") from None
        self._setup()

    def load_filenames(self) -> None:
        """Load the filenames and labels from the dataset directory."""
        self._load_filenames = True
        if self._labels is None:
            self._labels = []
        for folder_name in self._folder_names:
            folder_path = self._local_path / folder_name
            for file in folder_path.iterdir():
                if file.is_file():
                    self._filenames.append(file.relative_to(self._local_path))
                    if len(self._data_label_mapping.items()) > 0:
                        # Perform label mapping
                        self._labels.append(self._data_label_mapping[folder_name])
                    else:
                        # FIXME: This is not ideal, but it's a quick fix
                        # We should the class name/id and not the folder name
                        self._labels.append(folder_name)

    def _setup(self) -> None:
        """Check if the URI is a local path or a remote URI and prepare the dataset.

        Remote URIs can look like local paths, but they come in a YAML
        which also has a field `annotation_project` which is used to download the images.

        Local URIs are processed by _prepare_local_dataset()
        Remote URIs are processed by _prepare_remote_dataset()

        In both cases, a local_path will exist, which contains the samples.
        If we have a local URI and no folder exists, we raise an error.
        """
        if self._uri is None:
            errmsg = "Warning no URI is set up. Local Path can't be created"
            raise ValueError(errmsg)

        # Define _local_path from _uri under _data_path
        # Remote URIs might contain "://", which is replace by "/" to create a local folder
        self._local_path = self._data_path / pathlib.Path(str(self._uri).replace("://", "/"))

        if self._local_path.is_file():
            errmsg = "Invalid URI: it cannot be a file."
            raise ValueError(errmsg) from None

        # Check if our data is local/remote
        is_local = self._annotation_project_config is None

        # If _local_path does not exist, it must be a remote URI; create a local folder
        if not self._local_path.exists():
            if is_local:
                errmsg = f"The specified URI does not exist: {self._local_path}"
                raise FileNotFoundError(errmsg) from None
            else:
                self._local_path.mkdir(parents=True, exist_ok=True)

        # Route to dataset preparation: local or remote
        if is_local:
            self._prepare_local_dataset()
        else:
            self._prepare_remote_dataset()

    def _prepare_local_dataset(self) -> None:
        self._folder_names = [d.name for d in self._local_path.iterdir() if d.is_dir()]
        if not self._folder_names:
            raise ValueError("The directory does not contain any subfolders (classes).")

        # Check consistency folder-class-label
        # - If there are no data_label_mapping, use the folder names as labels
        if len(self._data_label_mapping) < 1:
            self._data_label_mapping = {folder_name: i for i, folder_name in enumerate(self._folder_names)}
        # - All data_label_mapping (mapping from config) must be in folder_names (directory list)
        for folder_name in self._data_label_mapping:
            if folder_name not in self._folder_names:
                raise ValueError(f"Folder name '{folder_name}' is not in the dataset. Check `data_label_mapping` and your dataset folder.")  # noqa: E501
        # - All folder_names (directory list) must be in data_label_mapping (mapping from config)
        # FIXME: Maybe that's not necessary? We might want to ignore some folders?
        for folder_name in self._folder_names:
            if folder_name not in self._data_label_mapping:
                raise ValueError(f"Folder name '{folder_name}' is not in the dataset. Check `data_label_mapping` and your dataset folder.")  # noqa: E501
        # - If there are no class names, use the folder names, but only if they introduce a new label value
        if self._class_names is None or len(self._class_names) < 1:
            self._class_names = []
            labels = set()
            for folder, label in self._data_label_mapping.items():
                if label not in labels:
                    self._class_names.append(folder)
                    labels.add(label)
        # - The number of class_names must match the number of values in data_label_mapping
        if len(self._class_names) != len(set(self._data_label_mapping.values())):
            raise ValueError("The number of class names must match the number of values in `data_label_mapping`.")

        # Traverse directories and collect filenames and their labels
        if self._load_filenames:
            self.load_filenames()

    def _prepare_remote_dataset(self) -> None:
        # Instantiate DataRepository
        if self._data_repository is None:
            if self._annotation_project_config is not None:
                self._data_repository = DataRepository(config=self._annotation_project_config)
            else:
                raise ValueError("Missing annotation_project config!") from None
        # Download data
        self._data_repository.download_data(debug=True)
        # Get local filepaths and annotated labels
        # Note:
        # - Each sample can have several labels (as in Label Studio)
        # - A label can be in the _data_label_mapping dictionary or not
        # BUT all that is already handled in get_local_data_filepaths_and_labels!
        # We simply get a flattened list of local files and their relevant labels :)
        filepaths, labels = self._data_repository.get_local_data_filepaths_and_labels(
            only_labels=self._data_label_mapping.keys())
        # Map labels
        self._filenames = []
        self._labels = []
        for filepath, label in zip(filepaths, labels):
            self._labels.append(self._data_label_mapping[label])
            self._filenames.append(filepath)

    @property
    def filenames(self) -> List[str]:  # noqa: D102
        return self._filenames

    @property
    def labels(self) -> List[str]:  # noqa: D102
        return self._labels

    @property
    def class_names(self) -> List[str]:  # noqa: D102
        return self._class_names

    def __getitem__(self, index: int) -> Tuple[str, str]:
        """Get the filename and label of the item at the specified index."""
        if self._load_filenames:
            if index >= len(self):
                raise IndexError("Index out of range.")
            return (str(self._filenames[index]), self._labels[index])
        else:
            raise ValueError("Filenames have not been loaded. Use load_filenames() first.")

    def __iter__(self) -> "Dataset":
        """Make the dataset iterable."""
        self._current_index = 0
        return self

    def __next__(self) -> Tuple[str, str]:
        """Get the next item in the dataset."""
        if self._current_index < len(self):
            result = self[self._current_index]
            self._current_index += 1
            return result
        else:
            raise StopIteration

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return len(self._filenames)


class ImageDataset(Dataset):
    """A dataset class for handling image data with options for resizing, format conversion, and channel order.

    The images are loaded lazily by default, but can be loaded into memory.
    The **kwargs are passed to the resize_image function (min_size, keep_aspect_ratio, etc.)

    Usage:
        ds = ImageDataset(yaml_path="path/to/dataset.yaml",
                          load_images=False,
                          image_format=ImageFormat.NUMPY,
                          channels=ChannelOrder.RGB)

        image, label = ds[0]  # Get the first item, e.g.: (np.ndarray, "class_1")

        ds.load_images()  # Load all images into memory
        images = ds.images  # Get all images as a list (np.ndarray or PIL.Image)
        labels = ds.labels  # Get all labels as a list

        # We can also use the static methods for image processing,
        # specially useful for inference:
        transformers = [ImageResizer(min_size=300, keep_aspect_ratio=True)]
        img_1 = ImageDataset.load_and_prepare_image("path/to/image.jpg", transformers=transformers)
        img_2 = ImageDataset.prepare_image(img_1, transformers=transformers)
    """
    def __init__(self,
                 uri: Optional[Union[str, pathlib.Path]] = None,
                 labels: Optional[List[str]] = None,
                 data_label_mapping: Optional[Dict] = None,
                 yaml_path: Optional[Union[str, pathlib.Path]] = None,
                 data_path: Optional[Union[str, pathlib.Path]] = None,
                 load_filenames: bool = True,
                 load_images: bool = False,
                 image_format: ImageFormat = ImageFormat.NUMPY,
                 channels: ChannelOrder = ChannelOrder.RGB,
                 transformers: Optional[List[DataTransformer]] = None) -> None:
        super().__init__(uri=uri,
                         labels=labels,
                         data_label_mapping=data_label_mapping,
                         yaml_path=yaml_path,
                         data_path=data_path, # converted to DATA_PATH if passed None
                         load_filenames=load_filenames)
        self._load_images = load_images
        self._image_format = image_format
        self._channels = channels
        self._transformers = transformers
        self._images: Union[List[Union[Image.Image, np.ndarray]], None] = None

        if self._load_images:
            self.load_images()

    def load_images(self) -> None:
        """Load all images into memory."""
        self._load_images = True
        self._images = [self.load_and_prepare_image(self._local_path / filename,
                                                    image_format=self._image_format,
                                                    channels=self._channels,
                                                    transformers=self._transformers) \
                        for filename in self._filenames]

    @staticmethod
    def load_and_prepare_image(image_path: Union[str, pathlib.Path],
                               image_format: ImageFormat = ImageFormat.NUMPY,
                               channels: ChannelOrder = ChannelOrder.RGB,
                               transformers: Optional[List[DataTransformer]] = None) -> Any:
        """Load and prepare an image based on the provided settings.

        This is a static method, i.e., it can be called without instantiating
        the class:

            img = ImageDataset.load_and_prepare_image(image_path="path/to/image.jpg")

        The static function makes sense for independent utility functions,
        which are not tied to the state of the class; this is necessary
        for the inference pipeline.
        """
        # FIXME: Is this the best way to handle this?
        # Load image using image_utils (e.g., PIL, OpenCV)
        image = load_image(image_path, format=image_format, channels=channels)
        if transformers is not None:
            for t in transformers:
                image = t.transform(image)
        return image

    @staticmethod
    def prepare_image(image: Union[Image.Image, np.ndarray],
                      image_format: ImageFormat = ImageFormat.NUMPY,
                      channels: ChannelOrder = ChannelOrder.RGB,
                      transformers: Optional[List[DataTransformer]] = None) -> Any:
        """Prepare an image for processing.

        This is a static method, i.e., it can be called without instantiating
        the class:

            img = ImageDataset.prepare_image(img)

        The static function makes sense for independent utility functions,
        which are not tied to the state of the class; this is necessary
        for the inference pipeline.
        """
        # FIXME: Is this the best way to handle this?
        return_image = image
        if isinstance(image, Image.Image):
            if image_format == ImageFormat.NUMPY:
                return_image = pil2numpy(image, out_channels=channels)
        elif isinstance(image, np.ndarray):
            if image_format == ImageFormat.PIL:
                return_image = numpy2pil(image, in_channels=channels)
        else:
            raise ValueError("Invalid image type. Must be either PIL.Image or np.ndarray.")

        if transformers is not None:
            for t in transformers:
                return_image = t.transform(return_image)

        return return_image

    def __iter__(self) -> "ImageDataset":
        """Make the dataset iterable."""
        self._current_index = 0
        return self

    def __getitem__(self, index: int) -> Tuple[Union[Image.Image, np.ndarray], str]:
        """Get the image and label of the item at the specified index."""
        if index >= len(self):
            raise IndexError("Index out of range.")
        if self._load_images:
            return (self._images[index], self._labels[index])
        else:
            # Watch out: this assumes that the filenames & labels have been loaded!
            image = self.load_and_prepare_image(self._local_path / self._filenames[index],
                                                image_format=self._image_format,
                                                channels=self._channels,
                                                transformers=self._transformers)
            return (image, self._labels[index])

    def __next__(self) -> Tuple[Union[Image.Image, np.ndarray], str]:
        """Get the next item in the dataset."""
        if self._current_index < len(self):
            result = self[self._current_index]
            self._current_index += 1
            return result
        else:
            raise StopIteration

    @property
    def images(self) -> Union[List[Union[Image.Image, np.ndarray]], None]:  # noqa: D102
        # If load_images() has NOT been called, it will return None.
        return self._images

