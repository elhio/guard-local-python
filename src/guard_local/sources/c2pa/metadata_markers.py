"""
Hunts for generative AI markers buried inside assertion data.

Assertion payloads are free-form. Writers can put whatever they like in them at any
depth. Adobe in particular records a `generativeAI` flag somewhere inside its own
assertions rather than as a standard action. Because of this, the only way to find it
is to walk the payload.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = ["contains_generative_ai_marker"]

#: Keys that declare generative AI involvement when they hold a truthy value.
_MARKER_KEYS = ("generativeai", "generative_ai")

#: The maximum depth to walk. This prevents a cyclic or absurdly nested payload from
#: running away.
_MAX_DEPTH = 6


def contains_generative_ai_marker(value: Any, depth: int = 0) -> bool:
    """
    Walk an assertion payload looking for a generative AI marker.

    Args:
        value: Any part of the data from an assertion.
        depth: How far into the payload the current traversal has reached.

    Returns:
        True if a marker key is found with a value that is not an explicit denial. A
        marker set to `False` or `None` is a statement that no AI was involved and is
        deliberately not treated as a hit.
    """
    if depth > _MAX_DEPTH or value is None:
        return False

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in _MARKER_KEYS:
                if nested is not False and nested is not None:
                    return True
            elif contains_generative_ai_marker(nested, depth + 1):
                return True
        return False

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(contains_generative_ai_marker(item, depth + 1) for item in value)

    return False
