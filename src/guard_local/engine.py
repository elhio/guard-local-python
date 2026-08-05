"""
The on-device detector handling the ONNX session, media pipeline, and scoring.

The model is a multi-task FastViT-T8 with one head per detection task. Each head is
trained with `BCEWithLogits`. Because every output is a raw logit that can be unbounded
and negative, the engine applies a sigmoid function to convert them into meaningful
probabilities.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort

from .exceptions import ModelLoadError, UnsupportedMediaError
from .media_utils import load_frames, to_tensor
from .models import SUPPORTED_MEDIA_TYPES, TASKS, VIDEO_MEDIA_TYPES

__all__ = ["LocalDetectorEngine"]

#: Where the bundled model sits inside the installed package.
_BUNDLED_MODEL = ("assets", "lens_tiny_v1", "onnx", "model_fp16.onnx")

#: The filename to look for when a directory is given instead of a file.
_MODEL_FILENAME = os.path.join("onnx", "model_fp16.onnx")

#: Read when no path is passed in, so the standalone engine honours the same knob the
#: Guard client exposes as ``GUARD_LOCAL_MODEL_PATH`` / ``local_model_path=``.
_MODEL_PATH_ENV = "GUARD_LOCAL_MODEL_PATH"

_AGGREGATORS = {"max", "mean"}


class LocalDetectorEngine:
    """
    Scores media on-device against the bundled detection model.

    Constructing this engine is lightweight and never touches the filesystem. The ONNX
    session is built on the first `analyze` call and reused afterwards. This lazy
    loading is deliberate. It ensures that model loading issues are raised during
    analysis rather than at initialization, preventing the main client from
    misinterpreting a corrupt model file as a missing installation.

    Example:
        ```python
        engine = LocalDetectorEngine()
        with open("photo.jpg", "rb") as handle:
            engine.analyze(handle.read(), "image/jpeg")
        ```
    """

    def __init__(
        self,
        model_path: Optional[Union[str, os.PathLike[str]]] = None,
        *,
        video_frames: int = 8,
        video_aggregate: str = "max",
    ) -> None:
        """
        Prepare an engine without loading the model into memory.

        Args:
            model_path: The `.onnx` file or a directory containing
                `onnx/model_fp16.onnx`. If omitted, this falls back to the
                `GUARD_LOCAL_MODEL_PATH` environment variable, and then to the model
                bundled with this package.
            video_frames: How many frames to sample from a video clip.
            video_aggregate: How to combine the per-frame scores of a video clip.
                Accepts `"max"` or `"mean"`. The default is `"max"`, which
                reports the worst frame to provide a conservative moderation reading.

        Raises:
            ValueError: If `video_frames` is below one or `video_aggregate` is not a
                known strategy.
        """
        if video_frames < 1:
            raise ValueError("video_frames must be at least 1")
        if video_aggregate not in _AGGREGATORS:
            known = ", ".join(sorted(_AGGREGATORS))
            raise ValueError(f"video_aggregate must be one of: {known}")

        self.model_path = model_path
        self.video_frames = video_frames
        self.video_aggregate = video_aggregate

        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._lock = threading.Lock()

    def analyze(
        self, data: bytes, media_type: str
    ) -> List[Dict[str, Union[str, float, None]]]:
        """
        Score media against every detection task.

        Args:
            data: The raw media bytes to analyze. Still images are scored as a
                single frame. Videos are sampled into multiple frames, scored
                individually, and then aggregated into a final result.
            media_type: The MIME type of the media.

        Returns:
            A list of results with one entry per task, always returned in the
            same order. Each entry contains a `label`, a `score` representing a
            probability from 0.0 to 1.0, and a `description`.

        Raises:
            ModelLoadError: If the model could not be found or loaded.
            UnsupportedMediaError: If the media type is not supported by this engine.
            MediaDecodeError: If the media type is supported but the bytes cannot
                be decoded.
        """
        if media_type not in SUPPORTED_MEDIA_TYPES:
            supported = ", ".join(sorted(SUPPORTED_MEDIA_TYPES))
            raise UnsupportedMediaError(
                f"{media_type!r} is not supported by the local engine. "
                f"Supported: {supported}."
            )

        max_frames = self.video_frames if media_type in VIDEO_MEDIA_TYPES else 1
        frames = load_frames(data, media_type, max_frames=max_frames)

        session, input_name = self._ensure_session()
        outputs = [task.output for task in TASKS]

        per_frame = [
            session.run(outputs, {input_name: to_tensor(frame)}) for frame in frames
        ]

        return [
            {
                "label": task.label,
                "score": self._aggregate(
                    [
                        _sigmoid(float(np.asarray(result[index]).reshape(-1)[0]))
                        for result in per_frame
                    ]
                ),
                "description": task.description,
            }
            for index, task in enumerate(TASKS)
        ]

    def _ensure_session(self) -> Tuple[ort.InferenceSession, str]:
        """
        Build the ONNX session on first use and cache it for reuse.

        Returns:
            The cached session and the name of its single input.

        Raises:
            ModelLoadError: If the model file is missing or the runtime refused it.
        """
        with self._lock:
            if self._session is None:
                path = self._resolve_model_path()
                try:
                    session = ort.InferenceSession(
                        path, providers=["CPUExecutionProvider"]
                    )
                except Exception as exc:
                    raise ModelLoadError(
                        f"Could not load the detection model at {path!r} ({exc})."
                    ) from exc
                self._session = session
                self._input_name = session.get_inputs()[0].name

        assert self._input_name is not None
        return self._session, self._input_name

    def _resolve_model_path(self) -> str:
        """
        Find the model by preferring the most explicit source provided.

        Returns:
            The path to an existing `.onnx` file.

        Raises:
            ModelLoadError: If an explicitly configured path does not exist, or the
                bundled model is missing from the installation.
        """
        for candidate, origin in (
            (self.model_path, "the configured model path"),
            (os.environ.get(_MODEL_PATH_ENV), f"${_MODEL_PATH_ENV}"),
        ):
            if not candidate:
                continue
            path = os.fspath(candidate)
            if os.path.isdir(path):
                path = os.path.join(path, _MODEL_FILENAME)
            if not os.path.isfile(path):
                raise ModelLoadError(f"No model file at {path!r} (from {origin}).")
            return path

        return self._bundled_model_path()

    @staticmethod
    def _bundled_model_path() -> str:
        """
        Locate the model shipped inside this package.

        Returns:
            The path to the bundled `.onnx` file.

        Raises:
            ModelLoadError: If the installation carries no bundled model.
        """
        from importlib.resources import files

        try:
            resource = files(__package__).joinpath(*_BUNDLED_MODEL)
            path: str = os.fspath(resource) if isinstance(resource, os.PathLike) else ""
            if not path:
                raise TypeError("the package is not installed as real files")
        except Exception as exc:
            raise ModelLoadError(
                f"The bundled detection model could not be located ({exc}). "
                f"Set ${_MODEL_PATH_ENV} to point at a model file."
            ) from exc

        if not os.path.isfile(path):
            raise ModelLoadError(
                f"The bundled detection model is missing from this installation "
                f"(expected at {path!r}). Reinstall guard-local-detector, or set "
                f"${_MODEL_PATH_ENV} to point at a model file."
            )
        return path

    def _aggregate(self, scores: Sequence[float]) -> float:
        """
        Combine one task's per-frame scores into the final reported score.

        Args:
            scores: The probabilities calculated for every frame on a single task.

        Returns:
            A probability between 0.0 and 1.0. Emitting a float rather than a
            percentage is important because the Guard client rescales by value.
            Returning an integer like 1 would be misread as full confidence.
        """
        if not scores:
            return 0.0
        combined = (
            max(scores) if self.video_aggregate == "max" else sum(scores) / len(scores)
        )
        return min(1.0, max(0.0, combined))


def _sigmoid(logit: float) -> float:
    """
    Map a raw logit to a probability.

    Args:
        logit: The unbounded output from a single model head.

    Returns:
        The logistic value of the logit, clamped between 0.0 and 1.0.
    """

    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)
