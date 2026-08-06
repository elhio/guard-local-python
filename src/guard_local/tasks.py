"""
The detection tasks and media types mirroring the vocabulary of the cloud API.

Labels in this module act as a public API. The Guard client assigns local results a task
ID of `uuid5(namespace, label)`. Because of this, renaming a label will silently change
the ID that callers rely on. These labels match the exact tasks seeded by the backend.
This consistency allows developers to test against the local engine and then transition
to the cloud without writing conditional logic for which engine ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Tuple

from .detection import AI_GENERATED, EXPLICIT, VIOLENT

__all__ = [
    "IMAGE_MEDIA_TYPES",
    "SUPPORTED_MEDIA_TYPES",
    "TASKS",
    "VIDEO_MEDIA_TYPES",
    "Task",
]


@dataclass(frozen=True)
class Task:
    """
    One detection head of the model sharing the same name as the cloud API.

    Attributes:
        label: The plain text detection name and the seed used for the task ID.
        description: The detailed explanation returned by the cloud API.
        output: The name of the ONNX graph output carrying the logit for this head.
        category: The detection category this task reports, matched to the naming
            convention of the metadata and C2PA layers. This is the only place
            where the vocabulary of the cloud API and the signal vocabulary are
            tied together.
    """

    label: str
    description: str
    output: str
    category: str


#: The three heads in the exact order results are always reported.
TASKS: Tuple[Task, ...] = (
    Task(
        label="AI-Generated",
        description="Detect AI-generated or manipulated media",
        output="out_ai",
        category=AI_GENERATED,
    ),
    Task(
        label="Violence",
        description="Detect violent media",
        output="out_violence",
        category=VIOLENT,
    ),
    Task(
        label="Explicit",
        description="Detect sexually explicit media",
        output="out_nsfw",
        category=EXPLICIT,
    ),
)

#: Still images decoded to a single frame.
IMAGE_MEDIA_TYPES: FrozenSet[str] = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/heic",
    }
)

#: Video clips sampled into multiple frames before their scores are aggregated.
VIDEO_MEDIA_TYPES: FrozenSet[str] = frozenset(
    {
        "video/mp4",
        "video/webm",
        "video/quicktime",
    }
)

#: Every media type the Guard client can forward without filtering by engine capability.
SUPPORTED_MEDIA_TYPES: FrozenSet[str] = IMAGE_MEDIA_TYPES | VIDEO_MEDIA_TYPES
