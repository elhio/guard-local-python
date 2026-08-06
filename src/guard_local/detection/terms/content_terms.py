"""
Wording that describes violent or explicit content.

These terms are matched using a lenient case-insensitive substring search over flattened
metadata text, just like the AI term lists. The entries are deliberately specific and
mostly consist of multiple words. Short and ambiguous stems are intentionally omitted
because a fragment like `gore` inadvertently matches the word 'category', and `sex`
matches 'Sussex'.
"""

from __future__ import annotations

from typing import Tuple

__all__ = ["EXPLICIT_TERMS", "VIOLENCE_TERMS"]

VIOLENCE_TERMS: Tuple[str, ...] = (
    "graphic violence",
    "extreme violence",
    "violent content",
    "bloodshed",
    "gruesome",
    "mutilation",
    "mutilated",
    "dismemberment",
    "decapitation",
    "beheading",
    "massacre",
    "war crime",
    "torture",
    "brutal killing",
    "dead body",
    "corpse",
    "gunshot wound",
)

EXPLICIT_TERMS: Tuple[str, ...] = (
    "nsfw",
    "not safe for work",
    "sexually explicit",
    "explicit content",
    "explicit material",
    "pornographic",
    "pornography",
    "hardcore porn",
    "adult content",
    "adult material",
    "mature content",
    "nudity",
    "erotica",
    "erotic content",
    "xxx",
    "18+",
)
