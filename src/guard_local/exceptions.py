"""
Everything this package raises under one base class.

The Guard client calls `LocalDetectorEngine.analyze` without a standard try and except
block. Because of this, whatever escapes reaches the caller verbatim. Raising only
`GuardLocalError` subclasses keeps this surface small enough to catch easily. For
example, a corrupt JPEG must not surface as a raw `PIL.UnidentifiedImageError`, and a
bad model must not surface as a raw `onnxruntime` error.
"""

from __future__ import annotations

__all__ = [
    "GuardLocalError",
    "MediaDecodeError",
    "ModelLoadError",
    "UnsupportedMediaError",
]


class GuardLocalError(Exception):
    """
    Base class for every failure raised by the local engine.

    Callers that only want to know whether local detection worked can catch this single
    class and ignore the specific distinctions below it.
    """


class ModelLoadError(GuardLocalError):
    """
    The ONNX model could not be found, read, or loaded into a session.

    This is raised on first use rather than during construction. The client turns any
    constructor failure into a message saying the engine is not installed. This would be
    misleading advice for a model file that is merely missing or corrupt.
    """


class UnsupportedMediaError(GuardLocalError, ValueError):
    """
    The media type is not one this engine can score.

    This also subclasses `ValueError` to mirror the client `UnsupportedMediaTypeError`.
    This is because providing an unsupported media type is an invalid argument rather
    than an internal failure.
    """


class MediaDecodeError(GuardLocalError, ValueError):
    """
    The media type is supported but the bytes could not be decoded.

    This covers truncated or corrupt files as well as videos that yield no decodable
    frames.
    """
