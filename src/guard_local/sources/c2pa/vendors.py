"""
Matches manifest text against known generator names.

The vendor list itself is shared with the metadata layer to ensure both agree on what
counts as a generator. Only the matching logic lives here because the fields it
evaluates are messier than standard metadata tags. For example, Google tools often leave
the `softwareAgent` empty and identify themselves in a free text action description
instead, such as stating they applied a SynthID watermark. Similarly, the OpenAI image
model sets the `softwareAgent.name` to the opaque string `gpt-image` while its
`claim_generator_info` explicitly states it is the OpenAI Media Service API.
"""

from __future__ import annotations

from typing import Any, Optional

from guard_local.detection.terms import AI_GENERATOR_VENDORS

__all__ = ["matches_known_ai_vendor"]


def matches_known_ai_vendor(text: Any) -> Optional[str]:
    """
    Check whether a piece of manifest text names a known generator.

    Args:
        text: The text to check. Anything that is not a non-empty string will never
            match.

    Returns:
        The vendor fragment that matched so the caller can quote it as evidence, or
            `None` if no match is found.
    """
    if not text or not isinstance(text, str):
        return None
    lowered = text.lower()
    return next((vendor for vendor in AI_GENERATOR_VENDORS if vendor in lowered), None)
