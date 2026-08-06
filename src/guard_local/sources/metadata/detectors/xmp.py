"""
Reads signals out of the XMP packet.

XMP is where editing tools record their own name and history. This makes it the most
reliable place a generator identifies itself outside a signed C2PA manifest. The packet
is searched as raw XML text rather than parsed so a namespace this code has never heard
of is still scanned.
"""

from __future__ import annotations

from typing import List

from guard_local.detection import SignalMatch
from guard_local.detection.terms import AI_GENERATOR_VENDORS, AI_SOURCE_TERMS

from ..extract import RawImageMetadata
from ..signals import METADATA_SIGNALS
from ..text_match import find_matching_term
from .content_scan import scan_content_terms

__all__ = ["detect_xmp_signals"]


def detect_xmp_signals(metadata: RawImageMetadata) -> List[SignalMatch]:
    """
    Evaluate the XMP packet for generator names and generation parameters.

    Args:
        metadata: The parsed metadata segments.

    Returns:
        A list of every XMP signal that fired in catalogue order.
    """
    xmp = metadata.get("xmp")
    if not xmp:
        return []

    matches: List[SignalMatch] = []

    generator_vendor = find_matching_term(xmp, AI_GENERATOR_VENDORS)
    if generator_vendor:
        matches.append(
            SignalMatch.of(
                METADATA_SIGNALS.xmp_generator_vendor,
                f'XMP metadata matches known AI vendor "{generator_vendor}"',
                "metadata",
            )
        )

    source_term = find_matching_term(xmp, AI_SOURCE_TERMS)
    if source_term:
        matches.append(
            SignalMatch.of(
                METADATA_SIGNALS.xmp_source_terms,
                f'XMP metadata contains AI source/generation term "{source_term}"',
                "metadata",
            )
        )

    matches.extend(
        scan_content_terms(
            xmp,
            METADATA_SIGNALS.xmp_violent_content,
            METADATA_SIGNALS.xmp_explicit_content,
        )
    )

    return matches
