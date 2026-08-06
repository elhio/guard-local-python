"""
Analyzes the PNG header signal for signs of AI generation.

A PNG that survives a metadata strip still declares its dimensions. Diffusion models
typically work at a small set of fixed canvas sizes. This provides weak evidence but is
often the only thing left on an image that has been resaved.
"""

from __future__ import annotations

from typing import List

from guard_local.detection import SignalMatch
from guard_local.detection.terms import looks_like_typical_ai_dimension

from ..extract import RawImageMetadata
from ..signals import METADATA_SIGNALS

__all__ = ["detect_ihdr_signals"]


def detect_ihdr_signals(metadata: RawImageMetadata) -> List[SignalMatch]:
    """
    Check the declared dimensions of a PNG against known generator defaults.

    Args:
        metadata: The parsed metadata segments.

    Returns:
        A list containing the IHDR signal when the dimensions match exactly, or an empty
        list otherwise.
    """
    ihdr = metadata.get("ihdr")
    if not ihdr:
        return []

    width = ihdr.get("ImageWidth")
    height = ihdr.get("ImageHeight")
    if not looks_like_typical_ai_dimension(width, height):
        return []

    return [
        SignalMatch.of(
            METADATA_SIGNALS.ihdr_typical_ai_dimension,
            f"PNG dimensions {width}x{height} match a common AI generator output size",
            "metadata",
        )
    ]
