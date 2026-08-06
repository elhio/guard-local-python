"""
Decodes media bytes into the frames the model scores.

The engine receives only a buffer and a MIME type instead of a path or filename. Because
of this, all operations here work directly from bytes. This logic lives under the model
source because the model is the only component that requires pixel data. The provenance
and metadata sources read the same buffer without ever decoding it. This is why a
metadata-only engine executes in milliseconds. Frames leave this module EXIF-rotated and
converted to RGB. Alpha channels are dropped rather than composited, which represents
the first of the four steps documented in the transform module.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, List, Optional

import pillow_heif
from PIL import Image, ImageOps

from guard_local.exceptions import (
    GuardLocalError,
    MediaDecodeError,
    UnsupportedMediaError,
)
from guard_local.tasks import (
    IMAGE_MEDIA_TYPES,
    SUPPORTED_MEDIA_TYPES,
    VIDEO_MEDIA_TYPES,
)

__all__ = ["load_frames"]

# A clip whose container reports no duration must be walked frame by frame. We cap
# that walk so a long or malformed stream cannot turn one call into an unbounded decode.
_MAX_SEQUENTIAL_FRAMES = 900

# Pillow cannot decode HEIC unaided. Registering the opener here ensures HEIC
# flows through exactly the same still image path as everything else.
pillow_heif.register_heif_opener()


def load_frames(
    data: bytes, media_type: str, *, max_frames: int = 8
) -> List[Image.Image]:
    """
    Decode media bytes into the frames to be scored.

    Args:
        data: The raw media bytes.
        media_type: The MIME type of the media. This must be one of the
            supported types.
        max_frames: How many frames to sample from a video. This is ignored
            for still images which always yield exactly one frame.

    Returns:
        A list containing one RGB image for a still image, or up to
        `max_frames` images for a video.

    Raises:
        UnsupportedMediaError: If the media type is not supported by the engine.
        MediaDecodeError: If the bytes could not be decoded or if a video
            yielded no frames.
    """
    if media_type in IMAGE_MEDIA_TYPES:
        return [_load_image(data)]
    if media_type in VIDEO_MEDIA_TYPES:
        return _load_video_frames(data, max_frames=max_frames)

    supported = ", ".join(sorted(SUPPORTED_MEDIA_TYPES))
    raise UnsupportedMediaError(
        f"{media_type!r} is not supported by the local engine. Supported: {supported}."
    )


def _load_image(data: bytes) -> Image.Image:
    """
    Decode a still image.

    Args:
        data: The raw image bytes.

    Returns:
        The first frame EXIF-rotated and converted to RGB.

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

    PyAV opens a file object directly from the buffer. This means the clip is demuxed
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

    Targets sit at the midpoint of each of the `max_frames` equal slices. This keeps the
    sample away from the leading and trailing frames. Those are often black or
    letterboxed title cards that do not accurately represent the main content of the
    clip.

    Seeking only moves backward. It lands on the keyframe preceding a target, meaning
    the target frame itself must be reached by decoding forward. Because the targets
    ascend chronologically, one seek to the first target followed by a continuous
    forward walk will reach all of them. This is much faster than seeking per target,
    which would repeatedly decode the same keyframe intervals.

    Args:
        container: The open PyAV container.
        stream: The first video stream.
        max_frames: How many frames to sample.

    Returns:
        The decoded frames in presentation order, returning one per target reached.
        This returns an empty list if the clip reports no usable duration. That tells
        the caller to use the sequential fallback method.
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
        A list of up to `max_frames` decoded frames.
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
        container: The container which is consulted when the stream itself does not
            report a duration.

    Returns:
        The duration in `stream.time_base` units, or `None` if neither the stream nor
        the container knows it.
    """
    if stream.duration is not None:
        return int(stream.duration)
    if container.duration is not None and stream.time_base:
        # `container.duration` is in microseconds regardless of the stream's time base
        return int(container.duration / 1_000_000 / stream.time_base)
    return None


def _to_rgb(image: Image.Image) -> Image.Image:
    """
    Drop everything except the three color channels.

    Args:
        image: An image in any mode.

    Returns:
        The same image converted to RGB. Alpha channels are discarded rather than
        composited onto a background.
    """
    return image if image.mode == "RGB" else image.convert("RGB")
