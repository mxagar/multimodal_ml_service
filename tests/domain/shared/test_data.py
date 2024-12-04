"""Tests for the src/domain/shared/data.py module.

Author: Mikel Sagardia
Date: 2024-12-04
"""
import pytest
import numpy as np
from PIL import Image

from src.domain.shared.data import (ImageResizer,
                                    ImageFormatter,
                                    ChannelOrder,
                                    ImageFormat,
                                    save_data_transformers,
                                    load_data_transformers,
                                    Dataset,
                                    ImageDataset)


## --- DataTransformer: DummyImageTransformer


@pytest.mark.parametrize(
    "scale, expected",
    [(1.0, 42085243.0), (2.0, 84170486.0), (0.5, 21042621.5)] # if sample dataset and image change, this changes...
)
def test_dummy_image_transformer_transform(dummy_image_transformer,
                                           sample_image_dataset,
                                           scale,
                                           expected):
    """Test the DummyImageTransformer with a sample image dataset."""
    dummy_image_transformer.set_params(scale=scale)
    sample_image_dataset.load_images()
    dummy_images = sample_image_dataset.images
    transformed_features = dummy_image_transformer.transform(dummy_images[0])
    assert transformed_features.shape == (1,), "Expected output shape should be (1,)."
    assert transformed_features[0] >= expected*0.9, f"Expected transformed feature value to be {expected}."


## --- DataTransformer: ImageResizer


@pytest.mark.parametrize(
    "min_size, keep_aspect_ratio",
    [(50, True), (50, False), (75, True), (75, False)]
)
def test_image_resizer_numpy_image(image_sharp_numpy, min_size, keep_aspect_ratio):
    """Test the ImageResizer with a NumPy image."""
    resizer = ImageResizer(min_size=min_size, keep_aspect_ratio=keep_aspect_ratio)
    resized_image = resizer.transform(image_sharp_numpy)

    assert isinstance(resized_image, np.ndarray), "Expected output to be a NumPy array."
    if keep_aspect_ratio:
        assert resized_image.shape[0] == min_size or resized_image.shape[1] == min_size, \
            "Expected one of the dimensions to be equal to min_size while keeping aspect ratio."
    else:
        assert resized_image.shape[0] == min_size and resized_image.shape[1] == min_size, \
            "Expected both dimensions to be equal to min_size without keeping aspect ratio."


def test_image_resizer_get_set_params():
    """Test getting and setting parameters of the ImageResizer."""
    resizer = ImageResizer(min_size=50, keep_aspect_ratio=True)
    params = resizer.get_params()
    assert params == {"min_size": 50, "keep_aspect_ratio": True}, \
        "Expected parameters to match initial settings."

    resizer.set_params(min_size=100, keep_aspect_ratio=False)
    params = resizer.get_params()
    assert params == {"min_size": 100, "keep_aspect_ratio": False}, \
        "Expected parameters to match updated settings."


def test_image_resizer_save_load_params(artifacts_path):
    """Test saving and loading parameters of the ImageResizer."""
    resizer = ImageResizer(min_size=50, keep_aspect_ratio=True)
    save_path = artifacts_path / "resizer_params.yaml"
    resizer.save_params(save_path)

    new_resizer = ImageResizer(min_size=0, keep_aspect_ratio=False)
    new_resizer.load_params(save_path)
    params = new_resizer.get_params()
    assert params == {"min_size": 50, "keep_aspect_ratio": True}, \
        "Expected loaded parameters to match saved settings."


## --- DataTransformer: ImageFormatter


def test_image_formatter_transform_numpy_to_pil(image_sharp_numpy):
    """Test the ImageFormatter with a NumPy image."""
    formatter = ImageFormatter(out_format=ImageFormat.PIL,
                               in_channels=ChannelOrder.RGB,
                               out_channels=ChannelOrder.RGB)
    formatted_image = formatter.transform(image_sharp_numpy)
    assert isinstance(formatted_image, Image.Image), "Expected output to be a PIL image."


def test_image_formatter_transform_pil_to_numpy():
    """Test the ImageFormatter with a PIL image."""
    image_pil = Image.new("RGB", (100, 100), color="white")
    formatter = ImageFormatter(out_format=ImageFormat.NUMPY,
                               in_channels=ChannelOrder.RGB,
                               out_channels=ChannelOrder.BGR)
    formatted_image = formatter.transform(image_pil)
    assert isinstance(formatted_image, np.ndarray), "Expected output to be a NumPy array."


def test_image_formatter_get_set_params():
    """Test getting and setting parameters of the ImageFormatter."""
    formatter = ImageFormatter(out_format=ImageFormat.PIL,
                               in_channels=ChannelOrder.RGB,
                               out_channels=ChannelOrder.BGR)
    params = formatter.get_params()
    assert params == {"out_format": ImageFormat.PIL.value,
                      "out_channels": ChannelOrder.BGR.value,
                      "in_channels": ChannelOrder.RGB.value}, \
        "Expected parameters to match initial settings."

    formatter.set_params(out_format=ImageFormat.NUMPY,
                         out_channels=ChannelOrder.HSV,
                         in_channels=ChannelOrder.BGR)
    params = formatter.get_params()
    assert params == {"out_format": ImageFormat.NUMPY.value,
                      "out_channels": ChannelOrder.HSV.value,
                      "in_channels": ChannelOrder.BGR.value}, \
        "Expected parameters to match updated settings."


## --- save/load_data_transformers()


def test_save_load_data_transformers(artifacts_path):
    """Test saving and loading a list of ImageResizers and ImageFormatters."""
    # Create a list of two differently parametrized ImageResizers
    transformers = [
        ImageResizer(min_size=50, keep_aspect_ratio=True),
        ImageFormatter(out_format=ImageFormat.NUMPY, out_channels=ChannelOrder.HSV, in_channels=ChannelOrder.RGB)
    ]

    save_path = artifacts_path / "multiple_transformers.yaml"
    save_data_transformers(transformers, save_path)

    # Load the parameters into new transformers
    new_transformers = [
        ImageResizer(min_size=0, keep_aspect_ratio=False),
        ImageFormatter(out_format=ImageFormat.PIL, out_channels=ChannelOrder.RGB, in_channels=ChannelOrder.HSV)
    ]
    load_data_transformers(new_transformers, save_path)

    # Check if the loaded parameters match the saved ones
    assert new_transformers[0].get_params() == {"min_size": 50, "keep_aspect_ratio": True}, \
        "Expected loaded parameters to match saved settings for first transformer."
    assert new_transformers[1].get_params() == {"out_format": ImageFormat.NUMPY.value,
                                                "out_channels": ChannelOrder.HSV.value,
                                                "in_channels": ChannelOrder.RGB.value}, \
        "Expected loaded parameters to match saved settings for second transformer."


def test_save_load_single_data_transformer(artifacts_path):
    """Test saving and loading a single ImageResizer."""
    # Create a list with a single ImageResizer
    transformers = [ImageResizer(min_size=75, keep_aspect_ratio=True)]
    save_path = artifacts_path / "single_transformer.yaml"
    save_data_transformers(transformers, save_path)

    # Load the parameters into a new transformer
    new_transformers = [ImageResizer(min_size=0, keep_aspect_ratio=False)]
    load_data_transformers(new_transformers, save_path)

    # Check if the loaded parameters match the saved ones
    assert new_transformers[0].get_params() == {"min_size": 75, "keep_aspect_ratio": True}, \
        "Expected loaded parameters to match saved settings for the transformer."


def test_save_load_empty_data_transformers(artifacts_path):
    """Test saving and loading an empty list of ImageResizers."""
    # Create an empty list of transformers
    transformers = []
    save_path = artifacts_path / "empty_transformers.yaml"
    save_data_transformers(transformers, save_path)

    # Load the parameters into an empty list of transformers
    new_transformers = []
    load_data_transformers(new_transformers, save_path)

    # Check if the loaded list is still empty
    assert len(new_transformers) == 0, "Expected no transformers to be loaded."


## --- Dataset


def test_dataset_initialization_from_uri(dataset_uri, data_path):
    """Test the initialization of a Dataset from a URI."""
    dataset = Dataset(uri=dataset_uri, data_path=data_path)
    assert dataset._uri == dataset_uri, "Expected URI to be set correctly."
    assert dataset._local_path.exists(), "Expected the dataset local path to exist."


def test_dataset_initialization_from_yaml(dataset_yaml_path, data_path):
    """Test the initialization of a Dataset from a YAML configuration."""
    dataset = Dataset(yaml_path=dataset_yaml_path, data_path=data_path)
    assert dataset._yaml_path == dataset_yaml_path, "Expected YAML path to be set correctly."
    assert dataset._uri is not None, "Expected dataset URI to be set from the YAML configuration."
    assert dataset._local_path.exists(), "Expected dataset local path to exist."


def test_dataset_load_filenames(sample_dataset):
    """Test loading filenames and labels into a Dataset."""
    assert len(sample_dataset.filenames) > 0, "Expected filenames to be loaded."
    assert len(sample_dataset.labels) > 0, "Expected labels to be loaded."


def test_dataset_get_item(sample_dataset):
    """Test getting an item from a Dataset."""
    #sample_dataset.load_filenames()
    filename, label = sample_dataset[0]
    assert isinstance(filename, str), "Expected filename to be a string."
    assert isinstance(label, (str, int)), "Expected label to be a string or an int."


def test_dataset_iteration(sample_dataset):
    """Test iterating over a Dataset."""
    #sample_dataset.load_filenames()
    count = 0
    for item in sample_dataset:
        assert isinstance(item, tuple), "Expected each item to be a tuple (filename, label)."
        count += 1
    assert count == len(sample_dataset), "Expected the iteration to cover all items."


def test_dataset_setup_from_yaml(dataset_yaml_path, data_path):
    """Test setting up a Dataset from a YAML configuration."""
    dataset = Dataset(data_path=data_path)
    dataset.setup_from_yaml(dataset_yaml_path)
    assert dataset._uri is not None, "Expected URI to be set after setting up from YAML."
    assert dataset._local_path.exists(), "Expected the dataset local path to exist after YAML setup."


def test_dataset_invalid_uri(data_path):
    """Test initializing a Dataset with an invalid URI."""
    with pytest.raises(FileNotFoundError):
        Dataset(uri="invalid/path/to/dataset", data_path=data_path)


def test_dataset_invalid_yaml(data_path):
    """Test initializing a Dataset with an invalid YAML configuration."""
    with pytest.raises(FileNotFoundError):
        Dataset(yaml_path="invalid/path/to/config.yaml", data_path=data_path)


def test_dataset_index_out_of_range(sample_dataset):
    """Test accessing an out-of-range index in a Dataset."""
    #sample_dataset.load_filenames()
    with pytest.raises(IndexError):
        _ = sample_dataset[len(sample_dataset) + 1]


## --- ImageDataset


@pytest.mark.parametrize(
    "load_images",
    [True, False]
)
def test_image_dataset_initialization(dataset_uri, data_path, load_images):
    """Test the initialization of an ImageDataset."""
    ds = ImageDataset(uri=dataset_uri,
                      data_path=data_path,
                      load_images=load_images)
    assert len(ds) > 0, "Expected the dataset to have at least one image."
    assert ds.class_names is not None, "Expected class names to be loaded."
    assert len(ds.filenames) > 0, "Expected filenames to be loaded."
    if load_images:
        assert ds.images is not None, "Expected images to be loaded into memory."
    else:
        assert ds.images is None, "Expected images to not be loaded into memory."


@pytest.mark.parametrize(
    "image_format, channels",
    [(ImageFormat.PIL, ChannelOrder.RGB), (ImageFormat.NUMPY, ChannelOrder.BGR)]
)
def test_image_dataset_get_item(dataset_uri, data_path, image_format, channels):
    """Test getting an item from an ImageDataset."""
    ds = ImageDataset(uri=dataset_uri,
                      data_path=data_path,
                      image_format=image_format,
                      channels=channels,
                      load_images=False)
    image, label = ds[0]
    if image_format == ImageFormat.PIL:
        assert isinstance(image, Image.Image), "Expected a PIL image."
    elif image_format == ImageFormat.NUMPY:
        assert isinstance(image, np.ndarray), "Expected a NumPy array."
        assert image.shape[2] == 3, "Expected 3 channels for the output image."
    assert isinstance(label, (str, int)), "Expected the label to be a string or an int."


@pytest.mark.parametrize(
    "image_format, channels",
    [(ImageFormat.PIL, ChannelOrder.RGB), (ImageFormat.NUMPY, ChannelOrder.BGR)]
)
def test_image_dataset_load_and_prepare_image(image_sharp_path, image_format, channels, sample_transformers):
    """Test loading and preparing an image from an ImageDataset."""
    image = ImageDataset.load_and_prepare_image(image_path=image_sharp_path,
                                                image_format=image_format,
                                                channels=channels,
                                                transformers=sample_transformers)
    if image_format == ImageFormat.PIL:
        assert isinstance(image, Image.Image), "Expected a PIL image."
    elif image_format == ImageFormat.NUMPY:
        assert isinstance(image, np.ndarray), "Expected a NumPy array."
        assert image.shape[2] == 3, "Expected 3 channels for the output image."


@pytest.mark.parametrize(
    "image_format, channels",
    [(ImageFormat.PIL, ChannelOrder.RGB), (ImageFormat.NUMPY, ChannelOrder.BGR)]
)
def test_image_dataset_prepare_image(image_sharp_pil, image_format, channels, sample_transformers):
    """Test preparing an image from an ImageDataset."""
    prepared_image = ImageDataset.prepare_image(image_sharp_pil,
                                                image_format=image_format,
                                                channels=channels,
                                                transformers=sample_transformers)
    if image_format == ImageFormat.PIL:
        assert isinstance(prepared_image, Image.Image), "Expected a PIL image."
    elif image_format == ImageFormat.NUMPY:
        assert isinstance(prepared_image, np.ndarray), "Expected a NumPy array."
        assert prepared_image.shape[2] == 3, "Expected 3 channels for the output image."


def test_image_dataset_iteration(sample_image_dataset):
    """Test iterating over an ImageDataset."""
    ds = sample_image_dataset
    count = 0
    for image, label in ds:
        assert isinstance(image, (Image.Image, np.ndarray)), "Expected the image to be either PIL or NumPy format."
        assert isinstance(label, (str, int)), "Expected the label to be a string or an int."
        count += 1
    assert count == len(ds), "Expected the iteration to cover all images in the dataset."
