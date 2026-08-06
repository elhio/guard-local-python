"""
Reads signals out of the IPTC IIM block.

IPTC data is typically written by people and newsroom tooling rather than by cameras.
This makes it the primary segment where a human has described the content in words.
Because of this, it serves as the main source of violent and explicit evidence. It is
also a place where both generators and wire services state their origin outright.
"""

from __future__ import annotations

from typing import List

from guard_local.detection import SignalMatch
from guard_local.detection.terms import AI_GENERATOR_VENDORS, CAMERA_SOURCE_TERMS

from ..extract import RawImageMetadata
from ..signals import METADATA_SIGNALS
from ..text_match import find_matching_term
from .content_scan import scan_content_terms

__all__ = ["detect_iptc_signals"]


def detect_iptc_signals(metadata: RawImageMetadata) -> List[SignalMatch]:
    """
    Evaluate the IPTC fields for origin claims and content wording.

    Args:
        metadata: The parsed metadata segments.

    Returns:
        A list of every IPTC signal that fired in catalogue order.
    """
    iptc = metadata.get("iptc")
    if not iptc:
        return []

    matches: List[SignalMatch] = []

    generator_vendor = find_matching_term(iptc, AI_GENERATOR_VENDORS)
    if generator_vendor:
        matches.append(
            SignalMatch.of(
                METADATA_SIGNALS.iptc_generator_vendor,
                "IPTC caption/credit fields match known AI vendor "
                f'"{generator_vendor}"',
                "metadata",
            )
        )

    camera_term = find_matching_term(iptc, CAMERA_SOURCE_TERMS)
    if camera_term:
        matches.append(
            SignalMatch.of(
                METADATA_SIGNALS.iptc_camera_capture,
                "IPTC source/credit field contains camera-capture wording "
                f'"{camera_term}"',
                "metadata",
            )
        )

    matches.extend(
        scan_content_terms(
            iptc,
            METADATA_SIGNALS.iptc_violent_content,
            METADATA_SIGNALS.iptc_explicit_content,
        )
    )

    return matches
