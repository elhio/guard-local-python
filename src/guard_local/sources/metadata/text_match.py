"""
Searches loosely typed metadata for relevant words.

Metadata segments are arbitrarily nested bags of unknown types. The interesting strings
hide in different places depending on which tool wrote the file. Rather than teaching
every detector where to look, these helpers flatten a segment into a single blob of text
and search that string. Keys are flattened alongside values because a field named
`GenerativeAI` is just as telling as a field whose value explicitly says so.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

__all__ = ["find_matching_term", "flatten_to_searchable_text", "has_any_field"]

#: The maximum depth to walk a nested segment. This stops a cyclic or pathologically
#: nested structure from running away.
_MAX_DEPTH = 4


def flatten_to_searchable_text(value: Any, depth: int = 0) -> str:
    """
    Flatten a nested value into a single space-separated string of keys and values.

    Args:
        value: Any part of a parsed metadata segment.
        depth: How far into the structure the current traversal has reached.

    Returns:
        Every key and primitive reachable within the depth cap joined by spaces. Bytes
        are decoded as UTF-8 with replacement. This is necessary because XMP arrives
        raw and a single misencoded byte must not result in the loss of the entire
        packet.
    """
    if depth > _MAX_DEPTH or value is None:
        return ""

    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")

    if isinstance(value, (str, int, float, bool)):
        return str(value)

    if isinstance(value, Mapping):
        return " ".join(
            f"{key} {flatten_to_searchable_text(nested, depth + 1)}"
            for key, nested in value.items()
        )

    if isinstance(value, Iterable):
        return " ".join(flatten_to_searchable_text(item, depth + 1) for item in value)

    return ""


def find_matching_term(segment: Any, terms: Iterable[str]) -> Optional[str]:
    """
    Find the first of several terms mentioned anywhere in a metadata segment.

    Args:
        segment: The segment to search. A value of `None` or an empty segment will match
            nothing.
        terms: The terms to look for in priority order.

    Returns:
        The first term found, or `None` if there is no match. Matching is
        case-insensitive and evaluated by substring. This means a term will successfully
        match even if it is embedded inside a longer word.
    """
    if not segment:
        return None
    haystack = flatten_to_searchable_text(segment).lower()
    return next((term for term in terms if term.lower() in haystack), None)


def has_any_field(segment: Any, field_names: Iterable[str]) -> bool:
    """
    Check whether a segment carries any of several fields regardless of value.

    Args:
        segment: The segment to inspect. Only its top level is examined.
        field_names: The field names to look for.

    Returns:
        Whether at least one field is present when compared case-insensitively. This is
        used when the mere presence of a reading matters more than what it actually
        says, such as finding an aperture setting.
    """
    if not isinstance(segment, Mapping) or not segment:
        return False
    wanted = {name.lower() for name in field_names}
    return any(str(key).lower() in wanted for key in segment)
