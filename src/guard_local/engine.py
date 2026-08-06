"""
The on-device detector that evaluates and weighs multiple evidence sources.

Three sources analyze the same bytes and evaluate the same three categories. Each source
provides answers in the exact same format and operates entirely independently. This
module queries each source, merges the returned evidence, and formats the result exactly
as the Guard client expects.

1. A signed C2PA manifest chain acts as provenance. This chain cannot be edited without
   breaking its signature and often states directly whether an AI model generated the
   media.
2. Embedded metadata like EXIF, XMP, IPTC, ICC, JFIF, and PNG text provides additional
   context. Image generators routinely leave their names and prompts in these fields,
   just as cameras leave lens and exposure data.
3. A multi-task vision model looks at the pixels, which is all that remains when a file
   carries no metadata at all.

The strongest signal among the three determines the final result. These are not
independent samples that can simply be averaged. For example, a signed
`c2pa.ai_generated` action is not made less true just because a quiet vision model
disagrees with it.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

from .detection import CategoryResult, merge_category_results, passes_threshold
from .exceptions import UnsupportedMediaError
from .sources import C2paSource, MetadataSource, ModelSource, Source
from .sources.model import AGGREGATORS
from .tasks import SUPPORTED_MEDIA_TYPES, TASKS, Task

__all__ = ["LocalDetectorEngine"]


class LocalDetectorEngine:
    """
    Scores media on-device against provenance, metadata, and the bundled model.

    Constructing this engine is lightweight and never touches the filesystem. The ONNX
    session is built on the first `analyze` call that needs it and reused afterwards.
    This lazy loading is deliberate. It ensures that model loading issues are raised
    during analysis rather than at initialization, preventing the main client from
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
        use_model: bool = True,
        use_metadata: bool = True,
        use_c2pa: bool = True,
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
            use_model: Whether to run the vision model. Turning it off leaves the
                provenance and metadata sources, which need no ONNX session at all and
                answer in milliseconds.
            use_metadata: Whether to read embedded metadata.
            use_c2pa: Whether to read signed C2PA provenance.

        Raises:
            ValueError: If `video_frames` is below one, `video_aggregate` is not a
                known strategy, or every source has been turned off. The two video
                knobs are checked even when the model is off, so a typo is reported
                where it was written rather than on some later call.
        """
        if video_frames < 1:
            raise ValueError("video_frames must be at least 1")
        if video_aggregate not in AGGREGATORS:
            known = ", ".join(sorted(AGGREGATORS))
            raise ValueError(f"video_aggregate must be one of: {known}")
        if not (use_model or use_metadata or use_c2pa):
            raise ValueError(
                "At least one of use_model, use_metadata or use_c2pa must be enabled, "
                "otherwise there is nothing to detect with."
            )

        self.model_path = model_path
        self.video_frames = video_frames
        self.video_aggregate = video_aggregate
        self.use_model = use_model
        self.use_metadata = use_metadata
        self.use_c2pa = use_c2pa

        self.model = (
            ModelSource(
                model_path,
                video_frames=video_frames,
                video_aggregate=video_aggregate,
            )
            if use_model
            else None
        )

        # Ordered by trust, so a caller stepping through them reads the strongest
        # evidence first. The merge itself is order-independent.
        sources: List[Source] = []
        if use_c2pa:
            sources.append(C2paSource())
        if use_metadata:
            sources.append(MetadataSource())
        if self.model is not None:
            sources.append(self.model)
        self.sources = tuple(sources)

    def analyze(self, data: bytes, media_type: str) -> List[Dict[str, Any]]:
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
            probability from 0.0 to 1.0, a `description`, whether the task was
            `detected`, and the `matches` that argued for it.

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

        found = merge_category_results(
            source.analyze(data, media_type) for source in self.sources
        )

        return [self._report(task, found.get(task.category)) for task in TASKS]

    @staticmethod
    def _report(task: Task, evidence: Optional[CategoryResult]) -> Dict[str, Any]:
        """
        Phrase the merged evidence of one category as a task result.

        Args:
            task: The task being reported.
            evidence: What the sources found, or `None` when no source spoke to this
                task's category.

        Returns:
            One result entry with the `score` constrained to the unit interval.
            That specific range is critical because the Guard client rescales
            by value, meaning an integer `1` would be misread as full confidence
            rather than one percent.
        """
        confidence = evidence.confidence if evidence else 0.0
        matches = evidence.matches if evidence else []

        return {
            "label": task.label,
            "score": min(1.0, max(0.0, confidence / 100.0)),
            "description": task.description,
            "detected": passes_threshold(task.category, confidence),
            "matches": [match.to_dict() for match in matches],
        }
