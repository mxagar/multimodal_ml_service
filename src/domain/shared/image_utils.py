"""General image processing utilities.

This module provides functions to work with images, including:

- Loading and saving images.
- Converting images to and from Base64 strings.
- Changing the channel order and format of the images.
- Resizing images.
- Displaying images.

Author: Mikel Sagardia
Date: 2024-12-04
"""
import os
import base64
import io
from enum import Enum
from typing import Optional, Union
import pathlib

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class ChannelOrder(Enum):  # noqa: D101
    RGB = "RGB"
    BGR = "BGR"
    HSV = "HSV"


class ImageFormat(Enum): # noqa: D101
    PIL = "PIL"
    NUMPY = "NUMPY"


def base64_to_image(base64_string: str) -> Image:
    """Decodes a Base64-encoded string and returns an image.

    Args:
    ----
        base64_string (str): The Base64-encoded string.

    Returns:
    -------
        image (PIL.Image): A PIL Image object.

    """
    # Remove prefix
    if base64_string.startswith("data:image"):
        base64_string = base64_string.split(",")[1]

    # Decode the Base64 string
    image_data = base64.b64decode(base64_string)

    # Load image from bytes
    image = Image.open(io.BytesIO(image_data))

    return image


def image_to_base64(image_path: Union[pathlib.Path, str],
                    format: str = "JPEG"):  # noqa: A002
    """Converts an image to a Base64-encoded string.

    Args:
    ----
        image_path (str): The path to the image.
        format (str): The format of the image (e.g., 'JPEG', 'PNG'). Defaults to 'JPEG'.

    Returns:
    -------
        str: A Base64-encoded string of the image.

    """
    if isinstance(image_path, pathlib.Path):
        image_path = str(image_path)

    # Load the image using PIL
    image = Image.open(image_path)

    # Create BytesIO buffer to store the image data
    buffered = io.BytesIO()
    image.save(buffered, format=format)

    # Convert the image to a Base64 string
    base64_image_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Optional: add the prefix for the image format (MIME type)
    base64_image_with_prefix = f"data:image/{format.lower()};base64,{base64_image_string}"

    return base64_image_with_prefix


def numpy2pil(image: np.ndarray,
              in_channels: Optional[ChannelOrder] = ChannelOrder.RGB) -> Image.Image:
    """Watch out: data is copied! Output is forced to be RGB!"""
    if in_channels == ChannelOrder.BGR:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif in_channels == ChannelOrder.HSV:
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    return Image.fromarray(image)


def pil2numpy(image: Image.Image,
              out_channels: Optional[ChannelOrder] = ChannelOrder.RGB) -> np.ndarray:
    """Watch out: data is copied!"""
    image = np.array(image) # RGB
    if out_channels == ChannelOrder.BGR:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif out_channels == ChannelOrder.HSV:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    return image


def change_image_channels(image: Union[np.ndarray, Image.Image],
                          in_channels: ChannelOrder,
                          out_channels: ChannelOrder) -> Union[np.ndarray, Image.Image]:
    """Change the channel order of an image.

    Watch out:
    - data is copied!
    - PIL images are forced to be RGB!
    - if a PIL is passed and its out_channels != RGB, it's converted to NUMPY!
    """
    if in_channels == out_channels:
        return image
    elif isinstance(image, np.ndarray):
        # FIXME: Maybe double conversion is inefficient...
        return pil2numpy(numpy2pil(image, in_channels), out_channels)
    elif isinstance(image, Image.Image):
        if in_channels != ChannelOrder.RGB:
            # FIXME: Maybe add support for other channel orders
            raise ValueError("PIL images can have only RGB channel order.")
        return pil2numpy(image, out_channels)


def load_image(filepath: Union[str, pathlib.Path],
               format: Optional[ImageFormat] = ImageFormat.PIL,  # noqa: A002
               channels: Optional[ChannelOrder] = ChannelOrder.RGB) -> Image:
    """Load an image from the specified path in required format and channel order."""
    if isinstance(filepath, pathlib.Path):
        filepath = str(filepath)
    image = None

    try:
        if format == ImageFormat.PIL:
            image = Image.open(filepath)
        elif format == ImageFormat.NUMPY:
            image = cv2.imread(filepath) # BGR
            if channels == ChannelOrder.RGB:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif channels == ChannelOrder.HSV:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    except FileExistsError as e:
        raise e(f"The file at {filepath} does not exist.") from None

    return image


def resize_image(image: Union[Image.Image, np.ndarray],
                 min_size: int,
                 keep_aspect_ratio: bool = True) -> Union[Image.Image, np.ndarray]:
    """Resize an image to have a minimum size while optionally keeping the aspect ratio.

    Watch out: if we pass a numpy array, it is internally converted to PIL and back to numpy,
    so data is copied!

    Args:
        image (Union[Image.Image, np.ndarray]): The image to resize.
        min_size (int): The minimum size for the smallest dimension.
        keep_aspect_ratio (bool): Whether to keep the aspect ratio. Defaults to True.

    Returns:
        Union[Image.Image, np.ndarray]: The resized image.
    """
    resized_image = image
    if isinstance(resized_image, np.ndarray):
        resized_image = Image.fromarray(resized_image)

    # In PIL w & h are transposed as compared to OpenCV
    width, height = resized_image.size

    if keep_aspect_ratio:
        if width < height:
            new_width = min_size
            new_height = int(min_size * height / width)
        else:
            new_height = min_size
            new_width = int(min_size * width / height)
    else:
        new_width = new_height = min_size

    resized_image = resized_image.resize((new_width, new_height), Image.LANCZOS)

    if isinstance(image, np.ndarray): # original image
        return np.array(resized_image)
    return resized_image


def resize_images_in_folder(folder_path: Union[str, pathlib.Path],
                            output_folder: Union[str, pathlib.Path],
                            min_size: int,
                            keep_aspect_ratio: bool = True,
                            pattern: Optional[str] = "*") -> None:
    """Resize the images in a folder.

    Usage:
        resize_images_in_folder("path/to/input/folder", "path/to/output/folder", 300, pattern="*.png")
    """
    if isinstance(folder_path, pathlib.Path):
        folder_path = str(folder_path)
    if isinstance(folder_path, pathlib.Path):
        output_folder = str(output_folder)

    if not pathlib.Path(folder_path).exists():
        raise FileNotFoundError(f"The folder at {folder_path} does not exist.")

    if not pathlib.Path(output_folder).exists():
        pathlib.Path(output_folder).mkdir(parents=True)

    # Allowed image extensions
    allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

    # Get list of image paths based on the pattern and filter out non-image files
    image_paths = [
        str(pathlib.Path(folder_path) / f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(tuple(allowed_extensions)) or f.endswith(pattern)
    ]

    for img_path in tqdm(image_paths, desc="Resizing images"):
        try:
            # Open image using PIL
            img = Image.open(img_path)
            # Convert to RGB
            img = img.convert("RGB")

            # Resize
            img_resized = resize_image(img, min_size, keep_aspect_ratio)

            # Save the image in the output folder with the same name
            output_path = pathlib.Path(output_folder) / pathlib.Path(img_path).name
            img_resized.save(str(output_path))
        except Exception as e:
            print(f"Failed to process {img_path}. Reason: {e}")  # noqa: T201

    print("Finished resizing images.")  # noqa: T201


def display_image(image: Union[Image.Image, np.ndarray]) -> None:
    """Display an image using PIL."""
    if isinstance(image, np.ndarray):
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    elif isinstance(image, Image.Image):
        image.show()
