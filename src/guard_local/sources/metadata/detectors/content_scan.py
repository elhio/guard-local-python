"""
Scans for violent and explicit terms across every text-bearing segment.

IPTC, XMP, EXIF, and PNG text all carry free-form descriptions. The wording that matters
is the same in all four of these segments. Only the emitted signal differs. Because of
this, the caller passes the two signals belonging to its specific standard.
"""

from __future__ import annotations

from typing import Any, List

from guard_local.detection import Signal, SignalMatch
from guard_local.detection.terms import EXPLICIT_TERMS, VIOLENCE_TERMS

from ..text_match import find_matching_term

__all__ = ["scan_content_terms"]


def scan_content_terms(
    segment: Any, violent_signal: Signal, explicit_signal: Signal
) -> List[SignalMatch]:
    """
    Scan one segment for violent and explicit wording.

    Args:
        segment: The metadata segment to search.
        violent_signal: The violence signal to emit belonging to this standard.
        explicit_signal: The explicit signal to emit belonging to this standard.

    Returns:
        A match for each category whose wording was found. It returns nothing for the
        categories that were not found.
    """
    matches: List[SignalMatch] = []

    violent_term = find_matching_term(segment, VIOLENCE_TERMS)
    if violent_term:
        matches.append(
            SignalMatch.of(
                violent_signal,
                f'Metadata text mentions violent-content term "{violent_term}"',
                "metadata",
            )
        )

    explicit_term = find_matching_term(segment, EXPLICIT_TERMS)
    if explicit_term:
        matches.append(
            SignalMatch.of(
                explicit_signal,
                f'Metadata text mentions explicit-content term "{explicit_term}"',
                "metadata",
            )
        )

    return matches
