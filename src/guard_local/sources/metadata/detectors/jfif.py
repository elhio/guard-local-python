"""
Analyzes the JFIF segment signal.

The presence of an APP0 segment says very little on its own. Its low confidence score of
10 reflects this. It is recorded because it is one of the few structural facts about a
JPEG that survives every metadata stripper. This makes it worth showing in the evidence
list even when it decides nothing.
"""

from __future__ import annotations

from typing import List

from guard_local.detection import SignalMatch

from ..extract import RawImageMetadata
from ..signals import METADATA_SIGNALS

__all__ = ["detect_jfif_signals"]


def detect_jfif_signals(metadata: RawImageMetadata) -> List[SignalMatch]:
    """
    Report whether the file carries a JFIF segment.

    Args:
        metadata: The parsed metadata segments.

    Returns:
        A list containing the JFIF signal when an APP0 segment is present, or an empty
        list otherwise.
    """
    if not metadata.get("jfif"):
        return []

    return [
        SignalMatch.of(
            METADATA_SIGNALS.jfif_present,
            "JFIF (APP0) segment present",
            "metadata",
        )
    ]
