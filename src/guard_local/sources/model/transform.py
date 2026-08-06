"""
Converts a decoded frame into the tensor the model expects.

The transform is not a simple resize. It accurately reproduces the exact steps the model
was trained with. This is also the same pipeline the Guard browser extension uses
against the identical ONNX file. Scores are only comparable if the input pixels are
processed exactly as follows:

1. EXIF rotated and converted to RGB, with alpha dropped instead of composited.
2. Scaled so the longest edge is 256, truncating the shorter edge with `int()`.
3. Centered on a 256x256 canvas where padding replicates the nearest edge pixel.
4. Rescaled to a 0-1 range, normalized with ImageNet statistics, and laidout as NCHW
   `float32`.

The first step belongs to the decode module which hands over an already-rotated RGB
image. The rest happens here.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

__all__ = ["IMAGE_SIZE", "letterbox", "to_tensor"]

#: The fixed spatial input size of the model in pixels.
IMAGE_SIZE = 256

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Fit an image into a square canvas without distorting it.

    The longest edge is scaled to `size` and the shorter one is truncated using `int()`.
    This matches the rounding behavior of the training transform which is often one
    pixel off from a standard `round()` operation. The remaining space is padded by
    replicating the nearest edge pixel. This ensures the border contains no color that
    the image did not already have.

    Args:
        image: An RGB image.
        size: The edge length of the output canvas.

    Returns:
        A `uint8` array of shape `(size, size, 3)`.
    """
    width, height = image.size
    scale = size / max(width, height)
    scaled_width = max(1, int(width * scale))
    scaled_height = max(1, int(height * scale))

    resized = image.resize(
        (scaled_width, scaled_height), resample=Image.Resampling.BICUBIC
    )
    pixels = np.asarray(resized, dtype=np.uint8)
    if scaled_width == size and scaled_height == size:
        return pixels

    left = (size - scaled_width) // 2
    top = (size - scaled_height) // 2
    return np.pad(
        pixels,
        (
            (top, size - scaled_height - top),
            (left, size - scaled_width - left),
            (0, 0),
        ),
        mode="edge",
    )


def to_tensor(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Turn an image into the input tensor required by the model.

    Args:
        image: An RGB image.
        size: The edge length of the model's square input.

    Returns:
        A `float32` array of shape `(1, 3, size, size)`. The data type is critical here.
        Numpy would silently promote the entire tensor to `float64` if the normalization
        constants were not also explicitly typed as `float32`. The ONNX runtime would
        reject a `float64` tensor.
    """
    pixels = letterbox(image, size).astype(np.float32) / np.float32(255.0)
    pixels = (pixels - _MEAN) / _STD
    return np.expand_dims(pixels.transpose(2, 0, 1), axis=0)
