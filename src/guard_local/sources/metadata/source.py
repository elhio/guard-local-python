"""
The metadata source as the engine sees it.
"""

from __future__ import annotations

from typing import Dict

from guard_local.detection import CategoryResult

from .detect import detect_metadata_signals
from .extract import extract_image_metadata

__all__ = ["MetadataSource"]


class MetadataSource:
    """
    Reads the segments embedded in a media file and reports what they reveal.
    """

    #: The source name used to identify matches produced by this detector.
    name = "metadata"

    def analyze(self, data: bytes, media_type: str) -> Dict[str, CategoryResult]:
        """
        Read every metadata segment the media carries and evaluate it.

        Args:
            data: The raw media bytes.
            media_type: The MIME type of the media. Video containers are skipped here
                because the extraction relies on a still image decoder.

        Returns:
            A result for each category that a detector matched on. An empty mapping is
            returned if nothing is found. This is the common case and means very little
            on its own because metadata is the most easily stripped form of evidence.
        """
        return detect_metadata_signals(extract_image_metadata(data, media_type))
