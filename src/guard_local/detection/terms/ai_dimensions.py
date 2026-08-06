"""
Canvas sizes that diffusion models typically default to or are constrained to use.

This provides weak evidence on its own, which is why the signals built upon it carry low
confidence. Plenty of real photographs are cropped into squares, and a 1920x1080
resolution is a screenshot just as often as it is an AI generation. It earns its place
by combining with stronger signals to build a complete picture.
"""

from __future__ import annotations

from typing import Optional, Tuple

__all__ = ["TYPICAL_AI_DIMENSIONS", "looks_like_typical_ai_dimension"]

TYPICAL_AI_DIMENSIONS: Tuple[Tuple[int, int], ...] = (
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1536, 1536),
    (2048, 2048),
    (1024, 1792),
    (1792, 1024),
    (832, 1216),
    (1216, 832),
    (1344, 768),
    (768, 1344),
    (1152, 896),
    (896, 1152),
    (1920, 1080),
    (1080, 1920),
)


def looks_like_typical_ai_dimension(
    width: Optional[int], height: Optional[int]
) -> bool:
    """
    Check whether a size exactly matches a known generator default.

    Args:
        width: The image width in pixels, if it is known.
        height: The image height in pixels, if it is known.

    Returns:
        True if the pair matches one of the known sizes exactly. Missing or zero
        dimensions never match.
    """
    if not width or not height:
        return False
    return (width, height) in TYPICAL_AI_DIMENSIONS
