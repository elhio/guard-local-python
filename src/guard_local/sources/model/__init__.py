"""
Looks at the pixels themselves.

This is the only source that needs the media decoded. It is also the only one that still
has something to say when a file carries no metadata and no provenance at all. It is
the weakest kind of evidence here because a model gives an opinion whereas a signed
manifest gives a verifiable claim.
"""

from __future__ import annotations

from .decode import load_frames
from .session import MODEL_PATH_ENV, ModelSession
from .source import AGGREGATORS, ModelSource
from .transform import IMAGE_SIZE, letterbox, to_tensor

__all__ = [
    "AGGREGATORS",
    "IMAGE_SIZE",
    "MODEL_PATH_ENV",
    "ModelSession",
    "ModelSource",
    "letterbox",
    "load_frames",
    "to_tensor",
]
