"""Dataset service for loading and managing image datasets.

This module provides a service for loading and managing image datasets,
i.e., it represents a catalogue of any dataset used in the project.

Author: Mikel Sagardia
Date: 2024-12-04
"""
from src.core import CONFIG_PATH
from src.domain.shared.data import ImageDataset
from src.domain.shared.image_utils import ChannelOrder, ImageFormat


def load_blur_dataset(load_to_memory: bool = True) -> ImageDataset:
    """Loads the blur detection dataset."""
    blur_dataset_yaml = CONFIG_PATH / "blur_dataset_kaggle.yaml"

    # transformers = [ImageResizer(min_size=200, keep_aspect_ratio=True)] # noqa: ERA001
    transformers = None
    ds = ImageDataset(
        yaml_path=blur_dataset_yaml,
        load_images=False,
        image_format=ImageFormat.NUMPY,
        channels=ChannelOrder.RGB,
        transformers=transformers,
    )

    # Load all images into memory
    if load_to_memory:
        ds.load_images()

    return ds


class ImageDatasetManager:
    """Dataset manager for all the datasets necessary for training pipelines."""
    # TODO/FIXME: This class could be improved with the use of Singletons
    # Additionally, we can load/deload to/from memory the datasets
    def __init__(self, load_to_memory: bool = True) -> None:
        self._blur = load_blur_dataset(load_to_memory=load_to_memory)

    @property
    def blur(self) -> ImageDataset:  # noqa: D102
        return self._blur


# Initialize the image dataset manager
image_datasets = ImageDatasetManager(load_to_memory=False)
