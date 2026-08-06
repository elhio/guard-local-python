"""
Finds the model file and holds the ONNX session that reads it.

The session is built on first use rather than at construction. This laziness is
deliberate because it means a corrupt or missing model surfaces as a failed analysis
rather than a failed import. Otherwise, the Guard client would misread a failed import
as the local engine not being installed at all.
"""

from __future__ import annotations

import os
import threading
from typing import Optional, Tuple, Union

import onnxruntime as ort

from guard_local.exceptions import ModelLoadError

__all__ = ["MODEL_PATH_ENV", "ModelSession"]

#: Where the bundled model sits inside the installed package.
_BUNDLED_MODEL = ("assets", "lens_tiny_v1", "onnx", "model_fp16.onnx")

#: The filename to look for when a directory is given instead of a file.
_MODEL_FILENAME = os.path.join("onnx", "model_fp16.onnx")

#: Read when no path is passed in, so the standalone engine honors the same knob the
#: Guard client exposes as `GUARD_LOCAL_MODEL_PATH` / `local_model_path=`.
MODEL_PATH_ENV = "GUARD_LOCAL_MODEL_PATH"


class ModelSession:
    """
    A lazily built ONNX session that is safe to share across threads.

    Example:
        ```python
        session = ModelSession()
        runtime, input_name = session.ensure()  # doctest: +SKIP
        ```
    """

    def __init__(self, model_path: Optional[Union[str, os.PathLike[str]]] = None):
        """
        Record where the model should come from without going to look for it.

        Args:
            model_path: The `.onnx` file or a directory containing
                `onnx/model_fp16.onnx`. If omitted this falls back to the
                `GUARD_LOCAL_MODEL_PATH` environment variable and then to the model
                bundled with this package.
        """
        self.model_path = model_path

        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._lock = threading.Lock()

    def ensure(self) -> Tuple[ort.InferenceSession, str]:
        """
        Build the ONNX session on first use and cache it for reuse.

        Returns:
            The cached session and the name of its single input.

        Raises:
            ModelLoadError: If the model file is missing or the runtime refused it.
        """
        with self._lock:
            if self._session is None:
                path = self.resolve_path()
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

    def resolve_path(self) -> str:
        """
        Find the model by preferring the most explicit source provided.

        Returns:
            The path to an existing `.onnx` file.

        Raises:
            ModelLoadError: If an explicitly configured path does not exist or the
                bundled model is missing from the installation.
        """
        for candidate, origin in (
            (self.model_path, "the configured model path"),
            (os.environ.get(MODEL_PATH_ENV), f"${MODEL_PATH_ENV}"),
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
            resource = files("guard_local").joinpath(*_BUNDLED_MODEL)
            path: str = os.fspath(resource) if isinstance(resource, os.PathLike) else ""
            if not path:
                raise TypeError("the package is not installed as real files")
        except Exception as exc:
            raise ModelLoadError(
                f"The bundled detection model could not be located ({exc}). "
                f"Set ${MODEL_PATH_ENV} to point at a model file."
            ) from exc

        if not os.path.isfile(path):
            raise ModelLoadError(
                f"The bundled detection model is missing from this installation "
                f"(expected at {path!r}). Reinstall guard-local-detector, or set "
                f"${MODEL_PATH_ENV} to point at a model file."
            )
        return path
