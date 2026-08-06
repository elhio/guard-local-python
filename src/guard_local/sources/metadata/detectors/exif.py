"""
Reads signals out of EXIF data.

This is the richest segment and the only one that argues in both directions. The same
scan that finds a generator name in the Software tag also finds the lens and exposure
readings indicating a real camera captured the image.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from guard_local.detection import SignalMatch
from guard_local.detection.terms import (
    AI_GENERATOR_VENDORS,
    AI_SOURCE_TERMS,
    looks_like_typical_ai_dimension,
)

from ..camera_evidence import collect_camera_evidence, has_strong_camera_evidence
from ..extract import RawImageMetadata
from ..signals import METADATA_SIGNALS
from ..text_match import find_matching_term
from .content_scan import scan_content_terms

__all__ = ["detect_exif_signals"]


def _pick(segment: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    """
    Narrow a segment to the fields a signal declares it inspects.

    Args:
        segment: The segment to narrow.
        keys: The field names to keep.

    Returns:
        Only the requested fields that are present. Narrowing before searching keeps the
        evidence string honest about exactly where the match came from.
    """
    return {key: segment[key] for key in keys if segment.get(key) is not None}


def detect_exif_signals(metadata: RawImageMetadata) -> List[SignalMatch]:
    """
    Evaluate the EXIF segments for generation and capture evidence.

    Args:
        metadata: The parsed metadata segments.

    Returns:
        Every EXIF signal that fired in catalogue order.
    """
    matches: List[SignalMatch] = []

    # IFD0 carries the primary tags such as Make, Model and Software, while the EXIF
    # sub-directory carries the camera settings. Which one holds a given tag varies by
    # writer, so they are evaluated together.
    exif_and_ifd0 = {**metadata.get("ifd0", {}), **metadata.get("exif", {})}

    software_fields = _pick(
        exif_and_ifd0, METADATA_SIGNALS.exif_software_vendor.parameters
    )
    software_vendor = find_matching_term(software_fields, AI_GENERATOR_VENDORS)
    if software_vendor:
        matches.append(
            SignalMatch.of(
                METADATA_SIGNALS.exif_software_vendor,
                f'EXIF software field matches known AI vendor "{software_vendor}"',
                "metadata",
            )
        )

    comment_fields = _pick(
        exif_and_ifd0, METADATA_SIGNALS.exif_generation_parameters.parameters
    )
    source_term = find_matching_term(comment_fields, AI_SOURCE_TERMS)
    if source_term:
        matches.append(
            SignalMatch.of(
                METADATA_SIGNALS.exif_generation_parameters,
                "EXIF comment field contains generation-parameter term "
                f'"{source_term}"',
                "metadata",
            )
        )

    width = exif_and_ifd0.get("ExifImageWidth") or exif_and_ifd0.get("ImageWidth")
    height = exif_and_ifd0.get("ExifImageHeight") or exif_and_ifd0.get("ImageHeight")
    if looks_like_typical_ai_dimension(width, height):
        matches.append(
            SignalMatch.of(
                METADATA_SIGNALS.exif_typical_ai_dimension,
                f"Image dimensions {width}x{height} match a common AI generator "
                "output size",
                "metadata",
            )
        )

    evidence = collect_camera_evidence(metadata)
    if has_strong_camera_evidence(evidence):
        matches.append(
            SignalMatch.of(
                METADATA_SIGNALS.exif_camera_capture,
                f"Camera capture evidence: make/model={evidence.has_make_model}, "
                f"lens={evidence.has_lens}, "
                f"captureSettings={evidence.has_capture_settings}",
                "metadata",
            )
        )

    if evidence.has_gps:
        matches.append(
            SignalMatch.of(
                METADATA_SIGNALS.exif_gps,
                "GPS coordinates present in EXIF/GPS segment",
                "metadata",
            )
        )

    matches.extend(
        scan_content_terms(
            exif_and_ifd0,
            METADATA_SIGNALS.exif_violent_content,
            METADATA_SIGNALS.exif_explicit_content,
        )
    )

    return matches
