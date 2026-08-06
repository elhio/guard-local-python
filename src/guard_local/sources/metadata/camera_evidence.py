"""
Weighs the traces that a physical camera leaves behind.

No single trace proves a real capture. A generator can write any manufacturer name it
likes, and a stripped photo carries no metadata at all. Requiring two independent traces
is what turns a guessable field into solid evidence. This is the exact rule the browser
extension applies.
"""

from __future__ import annotations

from dataclasses import dataclass

from guard_local.detection.terms import (
    CAMERA_MAKE_MODEL_TERMS,
    CAPTURE_SETTING_FIELD_NAMES,
    GPS_FIELD_NAMES,
    LENS_FIELD_NAMES,
)

from .extract import RawImageMetadata
from .text_match import find_matching_term, has_any_field

__all__ = ["CameraEvidence", "collect_camera_evidence", "has_strong_camera_evidence"]


@dataclass(frozen=True)
class CameraEvidence:
    """
    Represents the independent signs of a real capture that a file carries.

    Attributes:
        has_make_model: A known manufacturer names itself in the Make or Model field.
        has_lens: Lens or optical parameters are recorded.
        has_capture_settings: Exposure mechanics such as shutter speed or aperture are
            present.
        has_gps: Real-world geographic coordinates are attached.
    """

    has_make_model: bool
    has_lens: bool
    has_capture_settings: bool
    has_gps: bool


def collect_camera_evidence(metadata: RawImageMetadata) -> CameraEvidence:
    """
    Look for every sign of a physical capture at once.

    Args:
        metadata: The parsed metadata segments.

    Returns:
        One flag for each kind of trace. IFD0 and the EXIF subdirectory are examined
        together because the specific directory that holds a given tag varies depending
        on the writer.
    """
    exif_and_ifd0 = {**metadata.get("ifd0", {}), **metadata.get("exif", {})}
    gps = metadata.get("gps", {})

    return CameraEvidence(
        has_make_model=find_matching_term(exif_and_ifd0, CAMERA_MAKE_MODEL_TERMS)
        is not None,
        has_lens=has_any_field(exif_and_ifd0, LENS_FIELD_NAMES),
        has_capture_settings=has_any_field(exif_and_ifd0, CAPTURE_SETTING_FIELD_NAMES),
        has_gps=has_any_field(gps, GPS_FIELD_NAMES) or bool(gps),
    )


def has_strong_camera_evidence(evidence: CameraEvidence) -> bool:
    """
    Decide whether the traces add up to a real capture.

    Args:
        evidence: The collected flags.

    Returns:
        Whether at least two independent traces are present.
    """
    return (
        sum(
            (
                evidence.has_make_model,
                evidence.has_lens,
                evidence.has_capture_settings,
                evidence.has_gps,
            )
        )
        >= 2
    )
