"""Tests for the src/domain/shared/image_utils.py module.

Author: Mikel Sagardia
Date: 2024-12-04
"""
import pytest
import numpy as np
from PIL import Image

from src.domain.shared.image_utils import (load_image,
                                           numpy2pil,
                                           pil2numpy,
                                           change_image_channels,
                                           resize_image,
                                           ImageFormat,
                                           ChannelOrder)


@pytest.mark.parametrize(
    "image_path", ["image_blur_path", "image_sharp_path", "image_empty_path"]
)
def test_load_image_pil(image_path, request):
    """Test loading an image as a PIL image."""
    image = load_image(request.getfixturevalue(image_path), format=ImageFormat.PIL)
    assert isinstance(image, Image.Image), "Expected a PIL image."


@pytest.mark.parametrize(
    "image_path", ["image_blur_path", "image_sharp_path", "image_empty_path"]
)
def test_load_image_numpy_rgb(image_path, request):
    """Test loading an image as a NumPy array with RGB channels."""
    image = load_image(request.getfixturevalue(image_path), format=ImageFormat.NUMPY, channels=ChannelOrder.RGB)
    assert isinstance(image, np.ndarray), "Expected a NumPy array."
    assert image.shape[2] == 3, "Expected 3 channels for RGB."
    if image_path == "image_empty_path":
        assert (image == 0).all(), "Expected black image."


def test_load_image_numpy_hsv(image_blur_path):
    """Test loading an image as a NumPy array with HSV channels."""
    image = load_image(image_blur_path, format=ImageFormat.NUMPY, channels=ChannelOrder.HSV)
    assert isinstance(image, np.ndarray), "Expected a NumPy array."
    assert image.shape[2] == 3, "Expected 3 channels for HSV."


def test_load_image_file_not_found(image_invalid_path):
    """Test loading an image that does not exist."""
    with pytest.raises(FileNotFoundError):
        load_image(image_invalid_path)


@pytest.mark.parametrize(
    "in_channels", [ChannelOrder.RGB, ChannelOrder.BGR, ChannelOrder.HSV]
)
def test_numpy2pil(in_channels):
    """Test converting a NumPy array to a PIL image."""
    # Create a dummy NumPy image (100x100, 3 channels)
    image_np = np.ones((100, 100, 3), dtype=np.uint8) * 255
    image_pil = numpy2pil(image_np, in_channels=in_channels)
    assert isinstance(image_pil, Image.Image), "Expected a PIL image."
    assert image_pil.mode == "RGB", "Expected image mode to be RGB."


@pytest.mark.parametrize(
    "out_channels", [ChannelOrder.RGB, ChannelOrder.BGR, ChannelOrder.HSV]
)
def test_pil2numpy(out_channels):
    """Test converting a PIL image to a NumPy array."""
    # Create a dummy PIL image (100x100, RGB)
    image_pil = Image.new("RGB", (100, 100), color="white")
    image_np = pil2numpy(image_pil, out_channels=out_channels)
    assert isinstance(image_np, np.ndarray), "Expected a NumPy array."
    assert image_np.shape[2] == 3, "Expected 3 channels for the output image."


def test_change_image_channels_numpy_to_numpy(image_blur_numpy):
    """Test changing the channels of a NumPy image."""
    formatted_image = change_image_channels(image_blur_numpy,
                                            in_channels=ChannelOrder.RGB,
                                            out_channels=ChannelOrder.HSV)
    assert isinstance(formatted_image, np.ndarray), "Expected a NumPy array."
    assert formatted_image.shape[2] == 3, "Expected 3 channels for the output image."


def test_change_image_channels_pil_to_numpy(image_blur_pil):
    """Test changing the channels of a PIL image."""
    formatted_image = change_image_channels(image_blur_pil,
                                            in_channels=ChannelOrder.RGB,
                                            out_channels=ChannelOrder.BGR)
    assert isinstance(formatted_image, np.ndarray), "Expected a NumPy array."
    assert formatted_image.shape[2] == 3, "Expected 3 channels for the output image."


def test_change_image_channels_no_change(image_blur_numpy):
    """Test changing the channels of a NumPy image to the same channels."""
    unchanged_image = change_image_channels(image_blur_numpy,
                                            in_channels=ChannelOrder.RGB,
                                            out_channels=ChannelOrder.RGB)
    assert unchanged_image is image_blur_numpy, \
        "Expected the original image to be returned without modification."


@pytest.mark.parametrize(
    "keep_aspect_ratio", [True, False]
)
def test_resize_image(keep_aspect_ratio):
    """Test resizing an image."""
    # Create a dummy PIL image (100x200)
    image_pil = Image.new("RGB", (100, 200), color="white")
    resized_image = resize_image(image_pil, min_size=50, keep_aspect_ratio=keep_aspect_ratio)

    assert isinstance(resized_image, Image.Image), "Expected a PIL image."
    if keep_aspect_ratio:
        assert resized_image.size[0] == 50 or resized_image.size[1] == 50,\
            "Expected one dimension to be equal to min_size."
    else:
        assert resized_image.size == (50, 50),\
            "Expected both dimensions to be equal to min_size."
