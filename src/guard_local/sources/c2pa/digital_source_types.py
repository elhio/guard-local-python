"""
The IPTC source type vocabulary that names an algorithmic origin.

These URIs are the closest thing to a standard declaration that a machine created the
asset. Matching is exact rather than by substring because these are controlled
vocabulary identifiers. A near miss represents a completely different claim rather than
a weaker one.
"""

from __future__ import annotations

from typing import Optional, Tuple

__all__ = ["AI_DIGITAL_SOURCE_TYPES", "is_ai_digital_source_type"]

#: See https://cv.iptc.org/newscodes/digitalsourcetype/
AI_DIGITAL_SOURCE_TYPES: Tuple[str, ...] = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/dataDrivenMedia",
    "http://c2pa.org/digitalsourcetype/trainedAlgorithmicData",
)


def is_ai_digital_source_type(value: Optional[str]) -> bool:
    """
    Check whether a source-type URI declares an algorithmic origin.

    Args:
        value: The `digitalSourceType` from an action, if it has one.

    Returns:
        Whether it exactly matches a known algorithmic source type.
    """
    return bool(value) and value in AI_DIGITAL_SOURCE_TYPES
