"""
Reads signals out of the embedded colour profile.

These signals are rare and never conclusive on their own. However, some exporters stamp
their own name into the profile description. This metadata often survives edits that
strip EXIF data, which makes it worth the minor computational cost to check.
"""

from __future__ import annotations

from typing import List

from guard_local.detection import SignalMatch
from guard_local.detection.terms import AI_GENERATOR_VENDORS

from ..extract import RawImageMetadata
from ..signals import METADATA_SIGNALS
from ..text_match import find_matching_term

__all__ = ["detect_icc_signals"]


def detect_icc_signals(metadata: RawImageMetadata) -> List[SignalMatch]:
    """
    Evaluate the ICC profile description for a generator name.

    Args:
        metadata: The parsed metadata segments.

    Returns:
        A list containing the ICC signal if a known vendor names itself in the profile,
        or an empty list otherwise.
    """
    icc = metadata.get("icc")
    if not icc:
        return []

    vendor = find_matching_term(icc, AI_GENERATOR_VENDORS)
    if not vendor:
        return []

    return [
        SignalMatch.of(
            METADATA_SIGNALS.icc_vendor_profile,
            f'ICC profile description matches known AI vendor "{vendor}"',
            "metadata",
        )
    ]
