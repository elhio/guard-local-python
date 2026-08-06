"""
Scores pixels with the bundled multi-task vision model.

The model is a FastViT-T8 with one head per task. Because each head is trained with
`BCEWithLogits`, every output is an unbounded logit that requires a sigmoid activation
to represent a probability. This is the only source that has an opinion about all three
categories and the only one that must decode the media first. It is also the only source
whose confidence is not a whole percent. Because of this, it reports the probability it
produced alongside a match rounded for display.
"""

from __future__ import annotations

import math
import os
from typing import Dict, FrozenSet, Optional, Sequence, Union

import numpy as np

from guard_local.detection import CategoryResult, SignalMatch, passes_threshold
from guard_local.tasks import TASKS, VIDEO_MEDIA_TYPES

from .decode import load_frames
from .session import ModelSession
from .transform import to_tensor

__all__ = ["AGGREGATORS", "ModelSource"]

#: Defines how the per-frame scores of a clip may be folded into one.
AGGREGATORS: FrozenSet[str] = frozenset({"max", "mean"})

#: The evidence direction a model score points in per category depending on whether it
#: fired.
_KINDS = {
    "aiGenerated": ("aiGenerated", "authentic"),
    "violent": ("violence", "safe"),
    "explicit": ("explicit", "safe"),
}

#: Maps how the model names itself in an evidence list per category.
_LABELS = {
    "aiGenerated": "Local AI Model",
    "violent": "Local Violence Model",
    "explicit": "Local Explicit Model",
}


class ModelSource:
    """
    Runs the on-device model and reports one result per category.

    Constructing this object is free of computational overhead. The ONNX session is
    built on the first `analyze` call and reused afterwards.
    """

    #: How matches from this source name themselves.
    name = "model"

    def __init__(
        self,
        model_path: Optional[Union[str, os.PathLike[str]]] = None,
        *,
        video_frames: int = 8,
        video_aggregate: str = "max",
    ) -> None:
        """
        Prepare the source without loading the model into memory.

        Args:
            model_path: Where to find the `.onnx` file. Refer to `ModelSession`.
            video_frames: How many frames to sample from a video clip.
            video_aggregate: How to combine the per-frame scores of a clip. This must
                be one of the values in `AGGREGATORS`.
        """
        self.video_frames = video_frames
        self.video_aggregate = video_aggregate
        self.session = ModelSession(model_path)

    def analyze(self, data: bytes, media_type: str) -> Dict[str, CategoryResult]:
        """
        Decode the media and score every frame against every task.

        Args:
            data: The raw media bytes.
            media_type: The MIME type of the media.

        Returns:
            One result per category. This always returns all three categories because
            the model has an opinion about every task regardless of how confident it is.

        Raises:
            ModelLoadError: If the model could not be found or loaded.
            UnsupportedMediaError: If the media type is not supported by this engine.
            MediaDecodeError: If the media type is supported but the bytes cannot be
                decoded.
        """
        max_frames = self.video_frames if media_type in VIDEO_MEDIA_TYPES else 1
        frames = load_frames(data, media_type, max_frames=max_frames)

        session, input_name = self.session.ensure()
        outputs = [task.output for task in TASKS]
        per_frame = [
            session.run(outputs, {input_name: to_tensor(frame)}) for frame in frames
        ]

        results: Dict[str, CategoryResult] = {}
        for index, task in enumerate(TASKS):
            score = self._aggregate(
                [
                    _sigmoid(float(np.asarray(result[index]).reshape(-1)[0]))
                    for result in per_frame
                ]
            )
            confidence = score * 100.0
            results[task.category] = CategoryResult(
                detected=passes_threshold(task.category, confidence),
                # Kept as the probability the model produced rather than the whole
                # percent the match displays. The browser extension rounds before
                # comparing sources; comparing at full precision picks the same winner
                # and spares the caller a rounding it never asked for.
                confidence=confidence,
                matches=[self._match(task.category, score)],
            )
        return results

    def _aggregate(self, scores: Sequence[float]) -> float:
        """
        Combine the per-frame scores of a single task into the final model score.

        Args:
            scores: The probabilities calculated for every frame on a single task.

        Returns:
            A probability between 0.0 and 1.0.
        """
        if not scores:
            return 0.0
        combined = (
            max(scores) if self.video_aggregate == "max" else sum(scores) / len(scores)
        )
        return min(1.0, max(0.0, combined))

    @classmethod
    def _match(cls, category: str, score: float) -> SignalMatch:
        """
        Record the model opinion as a piece of evidence.

        Args:
            category: The category the score belongs to.
            score: The model probability from 0.0 to 1.0.

        Returns:
            A match naming the model so a caller reading `matches` can see what the
            model contributed next to what the metadata did.
        """
        positive, safe = _KINDS[category]
        percent = round(score * 100)
        return SignalMatch(
            id=f"local-model-{category}",
            category=category,
            label=_LABELS[category],
            description="Inference result from the locally running on-device model",
            confidence=percent,
            kind=positive if score > 0.5 else safe,
            evidence=f"Model confidence score: {percent}%",
            source=cls.name,
        )


def _sigmoid(logit: float) -> float:
    """
    Map a raw logit to a probability.

    Args:
        logit: The unbounded output from a single model head.

    Returns:
        The logistic value of the logit clamped between 0.0 and 1.0.
    """
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)
