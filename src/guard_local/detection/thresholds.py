"""
The confidence each category must reach before it counts as detected.

AI generation is held to a higher bar than the other two categories because the metadata
evidence for it is far easier to fake or to leave behind by accident. In the future,
these thresholds will be optimized and will become customizable.
"""

from __future__ import annotations

from typing import Dict

from .types import AI_GENERATED, EXPLICIT, VIOLENT

__all__ = ["DETECTION_THRESHOLDS", "passes_threshold"]

#: The minimum confidence, from 0 to 100, that marks a category as detected.
DETECTION_THRESHOLDS: Dict[str, int] = {
    AI_GENERATED: 90,
    VIOLENT: 70,
    EXPLICIT: 70,
}


def passes_threshold(category: str, score: float) -> bool:
    """
    Decide whether a confidence is high enough to call the category detected.

    Args:
        category: The category being judged.
        score: Its confidence, from 0 to 100.

    Returns:
        Whether the score meets or exceeds the category's threshold. An unknown
        category never passes, so a typo cannot silently flag everything.
    """
    threshold = DETECTION_THRESHOLDS.get(category)
    return threshold is not None and score >= threshold
