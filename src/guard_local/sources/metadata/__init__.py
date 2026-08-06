"""
Reads what a media file says about itself.

Embedded metadata is the most computationally inexpensive evidence available. It
requires no vision model and no network requests. A generator that names itself in an
EXIF tag settles the question before a single pixel is examined. However, metadata is
also the most easily removed form of evidence. Because of this fragility, an absence of
signals here means nothing at all.
"""

from __future__ import annotations

from .camera_evidence import (
    CameraEvidence,
    collect_camera_evidence,
    has_strong_camera_evidence,
)
from .detect import detect_metadata_signals
from .extract import RawImageMetadata, extract_image_metadata
from .signals import METADATA_SIGNALS, MetadataSignal
from .source import MetadataSource
from .text_match import find_matching_term, flatten_to_searchable_text, has_any_field

__all__ = [
    "METADATA_SIGNALS",
    "CameraEvidence",
    "MetadataSignal",
    "MetadataSource",
    "RawImageMetadata",
    "collect_camera_evidence",
    "detect_metadata_signals",
    "extract_image_metadata",
    "find_matching_term",
    "flatten_to_searchable_text",
    "has_any_field",
    "has_strong_camera_evidence",
]
