"""
Reads signals out of PNG text chunks.

This detector reads more than the header of a PNG. Tools like Stable Diffusion, ComfyUI,
and Automatic1111 all write the prompt, sampler, steps, and seed of a generation into a
`parameters` text chunk. They write nothing to EXIF or XMP. Without this detector, a PNG
straight out of any of these tools looks like an ordinary picture.
"""

from __future__ import annotations

from typing import List

from guard_local.detection import SignalMatch
from guard_local.detection.terms import AI_GENERATOR_VENDORS, AI_SOURCE_TERMS

from ..extract import RawImageMetadata
from ..signals import METADATA_SIGNALS
from ..text_match import find_matching_term
from .content_scan import scan_content_terms

__all__ = ["detect_png_text_signals"]


def detect_png_text_signals(metadata: RawImageMetadata) -> List[SignalMatch]:
    """
    Evaluate the text chunks of a PNG for generation parameters and generator names.

    Args:
        metadata: The parsed metadata segments.

    Returns:
        A list of every PNG text signal that fired in catalogue order.
    """
    png_text = metadata.get("png_text")
    if not png_text:
        return []

    matches: List[SignalMatch] = []

    source_term = find_matching_term(png_text, AI_SOURCE_TERMS)
    if source_term:
        matches.append(
            SignalMatch.of(
                METADATA_SIGNALS.png_text_generation_parameters,
                f'PNG text chunk contains generation-parameter term "{source_term}"',
                "metadata",
            )
        )

    vendor = find_matching_term(png_text, AI_GENERATOR_VENDORS)
    if vendor:
        matches.append(
            SignalMatch.of(
                METADATA_SIGNALS.png_text_vendor,
                f'PNG text chunk matches known AI vendor "{vendor}"',
                "metadata",
            )
        )

    matches.extend(
        scan_content_terms(
            png_text,
            METADATA_SIGNALS.png_text_violent_content,
            METADATA_SIGNALS.png_text_explicit_content,
        )
    )

    return matches
