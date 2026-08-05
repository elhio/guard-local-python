"""
Decodes media bytes and converts them into the tensor the model expects.

The engine receives only a buffer and a MIME type instead of a path or filename. Because
of this, all operations here work directly from `bytes`.

The transform is not a simple resize. It accurately reproduces the exact steps the model
was trained with. This is also the same pipeline the Guard browser extension uses
against the identical ONNX file. Scores are only comparable if the input pixels are
processed exactly as follows:

1. EXIF rotated and converted to RGB, with alpha dropped instead of composited.
2. Scaled so the longest edge is 256, truncating the shorter edge with `int()`.
3. Centered on a 256x256 canvas where padding replicates the nearest edge pixel.
4. Rescaled to a 0-1 range, normalized with ImageNet statistics, and laid
   out as NCHW `float32`.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, List, Optional

import numpy as np
import pillow_heif
from PIL import Image, ImageOps

from .exceptions import GuardLocalError, MediaDecodeError, UnsupportedMediaError
from .models import IMAGE_MEDIA_TYPES, SUPPORTED_MEDIA_TYPES, VIDEO_MEDIA_TYPES

__all__ = ["IMAGE_SIZE", "letterbox", "load_frames", "to_tensor"]

#: The fixed spatial input size of the model in pixels.
IMAGE_SIZE = 256

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# A clip whose container reports no duration has to be walked frame by frame. Cap that
# walk so a long or malformed stream cannot turn one call into an unbounded decode.
_MAX_SEQUENTIAL_FRAMES = 900

# Pillow cannot decode HEIC unaided. Registering the opener here means HEIC then flows
# through exactly the same still image path as everything else.
pillow_heif.register_heif_opener()


def load_frames(
    data: bytes, media_type: str, *, max_frames: int = 8
) -> List[Image.Image]:
    """
    Decode media bytes into the frames to be scored.

    Args:
        data: The raw media bytes.
        media_type: The MIME type of the media. Must be one of the supported types.
        max_frames: How many frames to sample from a video. This is ignored
            for still images, which always yield exactly one frame.

    Returns:
        One RGB image for a still image, or up to `max_frames` images for a video.

    Raises:
        UnsupportedMediaError: If the media type is not supported by the engine.
        MediaDecodeError: If the bytes could not be decoded or a video yielded
            no frames.
    """
    if media_type in IMAGE_MEDIA_TYPES:
        return [_load_image(data)]
    if media_type in VIDEO_MEDIA_TYPES:
        return _load_video_frames(data, max_frames=max_frames)

    supported = ", ".join(sorted(SUPPORTED_MEDIA_TYPES))
    raise UnsupportedMediaError(
        f"{media_type!r} is not supported by the local engine. Supported: {supported}."
    )


def letterbox(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Fit an image into a square canvas without distorting it.

    The longest edge is scaled to `size` and the shorter one is truncated
    using `int()`. This matches the rounding behavior of the training transform,
    which is often one pixel off from a standard `round()` operation. The
    remaining space is padded by replicating the nearest edge pixel. This
    ensures the border contains no color that the image did not already have.

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
        A `float32` array of shape `(1, 3, size, size)`. The data type is
        critical here. Numpy would silently promote the entire tensor to
        `float64` if the normalization constants were not also explicitly
        `float32`, which the ONNX runtime would reject.
    """
    pixels = letterbox(image, size).astype(np.float32) / np.float32(255.0)
    pixels = (pixels - _MEAN) / _STD
    return np.expand_dims(pixels.transpose(2, 0, 1), axis=0)


def _load_image(data: bytes) -> Image.Image:
    """
    Decode a still image.

    Args:
        data: The raw image bytes.

    Returns:
        The first frame, EXIF-rotated and converted to RGB.

    Raises:
        MediaDecodeError: If the bytes do not represent a decodable image.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        # The browser decodes through `createImageBitmap`, which honours EXIF
        # orientation by default. Skipping this would score phone photos sideways.
        rotated = ImageOps.exif_transpose(image)
    except Exception as exc:
        raise MediaDecodeError(f"Could not decode the image ({exc}).") from exc

    return _to_rgb(rotated if rotated is not None else image)


def _load_video_frames(data: bytes, *, max_frames: int) -> List[Image.Image]:
    """
    Sample frames evenly across a video clip.

    PyAV opens a file object directly from the buffer, which means the clip is demuxed
    without ever writing anything to the disk.

    Args:
        data: The raw video bytes.
        max_frames: How many frames to sample.

    Returns:
        Up to `max_frames` RGB images in presentation order. A clip shorter than the
            requested sample count will yield fewer frames.

    Raises:
        MediaDecodeError: If the clip could not be demuxed, carries no video stream, or
            yielded no decodable frames.
    """
    # Deferred: ffmpeg's bindings are the most expensive import in the package, and a
    # caller that only ever scores stills should not pay for them.
    import av

    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    try:
        with av.open(io.BytesIO(data)) as container:
            stream = next(iter(container.streams.video), None)
            if stream is None:
                raise MediaDecodeError("The file carries no video stream.")
            # Let ffmpeg parallelise decoding; frame extraction is the slow half.
            stream.thread_type = "AUTO"

            frames = _seek_frames(container, stream, max_frames)
            if not frames:
                frames = _walk_frames(container, stream, max_frames)
    except GuardLocalError:
        raise
    except Exception as exc:
        raise MediaDecodeError(f"Could not decode the video ({exc}).") from exc

    if not frames:
        raise MediaDecodeError("The video yielded no decodable frames.")
    return frames


def _seek_frames(container: Any, stream: Any, max_frames: int) -> List[Image.Image]:
    """
    Sample at evenly spaced timestamps in a single forward pass.

    Targets sit at the midpoint of each of the `max_frames` equal slices.
    This keeps the sample away from the leading and trailing frames, which
    are often black or letterboxed title cards that do not accurately represent
    the main content of the clip.

    Seeking only moves backward. It lands on the keyframe preceding a target,
    so the target frame itself must be reached by decoding forward. Because
    the targets ascend chronologically, one seek to the first target followed
    by a continuous forward walk will reach all of them. This is much faster
    than seeking per target, which would repeatedly decode the same keyframe
    intervals.

    Args:
        container: The open PyAV container.
        stream: The first video stream.
        max_frames: How many frames to sample.

    Returns:
        The decoded frames in presentation order, returning one per target reached. This
        returns an empty list if the clip reports no usable duration, which tells the
        caller to use the sequential fallback method.
    """
    duration = _stream_duration(stream, container)
    if duration is None or duration <= 0:
        return []

    start = stream.start_time or 0
    pending = [
        start + int(duration * (index + 0.5) / max_frames)
        for index in range(max_frames)
    ]

    try:
        container.seek(pending[0], stream=stream)
    except Exception:
        return []

    frames: List[Image.Image] = []
    for decoded, frame in enumerate(container.decode(stream)):
        if decoded >= _MAX_SEQUENTIAL_FRAMES:
            break
        if frame.pts is None:
            continue
        # A frame can satisfy several targets at once. This happens if a clip is shorter
        # than the sample count, or if a final frame overshoots the remaining targets.
        # We only score it once.
        reached = False
        while pending and frame.pts >= pending[0]:
            pending.pop(0)
            reached = True
        if reached:
            frames.append(_to_rgb(frame.to_image()))
        if not pending:
            break

    return frames


def _walk_frames(container: Any, stream: Any, max_frames: int) -> List[Image.Image]:
    """
    Sample by decoding forwards for clips reporting no duration.

    This method keeps every `stride` frame and doubles the stride whenever the sample
    overflows. This effectively spreads the result across a stream of unknown length in
    a single pass.

    Args:
        container: The open PyAV container.
        stream: The first video stream.
        max_frames: How many frames to sample.

    Returns:
        Up to `max_frames` decoded frames.
    """
    with contextlib.suppress(Exception):
        container.seek(0, stream=stream)

    kept: List[Image.Image] = []
    stride = 1
    for index, frame in enumerate(container.decode(stream)):
        if index >= _MAX_SEQUENTIAL_FRAMES:
            break
        if index % stride:
            continue
        kept.append(_to_rgb(frame.to_image()))
        if len(kept) > max_frames:
            kept = kept[::2]
            stride *= 2

    return kept[:max_frames]


def _stream_duration(stream: Any, container: Any) -> Optional[int]:
    """
    Find the duration of a clip using the native time base of the stream.

    Args:
        stream: The video stream.
        container: The container, which is consulted when the stream itself
            does not report a duration.

    Returns:
        The duration in `stream.time_base` units, or `None` if neither the
        stream nor the container knows it.
    """
    if stream.duration is not None:
        return int(stream.duration)
    if container.duration is not None and stream.time_base:
        # `container.duration` is in microseconds regardless of the stream's time base.
        return int(container.duration / 1_000_000 / stream.time_base)
    return None


def _to_rgb(image: Image.Image) -> Image.Image:
    """
    Drop everything except the three color channels.

    Args:
        image: An image in any mode.

    Returns:
        The same image converted to RGB. Alpha channels are discarded rather
        than composited onto a background, mirroring the browser pipeline.
    """
    return image if image.mode == "RGB" else image.convert("RGB")
