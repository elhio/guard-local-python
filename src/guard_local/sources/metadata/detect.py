"""
Runs every metadata detector and collects the final verdict.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from guard_local.detection import (
    CategoryResult,
    SignalMatch,
    bucket_matches_by_category,
)

from .detectors import (
    detect_exif_signals,
    detect_icc_signals,
    detect_ihdr_signals,
    detect_iptc_signals,
    detect_jfif_signals,
    detect_png_text_signals,
    detect_xmp_signals,
)
from .extract import RawImageMetadata

__all__ = ["METADATA_SIGNAL_DETECTORS", "detect_metadata_signals"]

#: The detectors, run in order. Order affects only the sequence matches are found in,
#: since the verdict is decided by confidence rather than by position.
METADATA_SIGNAL_DETECTORS: Tuple[
    Callable[[RawImageMetadata], List[SignalMatch]], ...
] = (
    detect_exif_signals,
    detect_xmp_signals,
    detect_iptc_signals,
    detect_icc_signals,
    detect_jfif_signals,
    detect_ihdr_signals,
    detect_png_text_signals,
)


def detect_metadata_signals(
    metadata: RawImageMetadata,
) -> Dict[str, CategoryResult]:
    """
    Run every metadata detector and group what they found.

    Args:
        metadata: The parsed metadata segments.

    Returns:
        A result for each category that a detector evaluated. Categories with no matches
        are omitted.
    """
    matches = [
        match for detect in METADATA_SIGNAL_DETECTORS for match in detect(metadata)
    ]
    return bucket_matches_by_category(matches)
