"""
The provenance source as seen by the detection engine.
"""

from __future__ import annotations

from typing import Dict

from guard_local.detection import AI_GENERATED, CategoryResult

from .detect import detect_ai_generation
from .reader import read_manifest_chain

__all__ = ["C2paSource"]


class C2paSource:
    """
    Reads a signed manifest chain and reports what it admits to.
    """

    #: The source name used to identify matches produced by this detector.
    name = "c2pa"

    def analyze(self, data: bytes, media_type: str) -> Dict[str, CategoryResult]:
        """
        Read and evaluate the provenance attached to the media, if any exists.

        Args:
            data: The raw media bytes.
            media_type: The MIME type of the media, which tells the reader how
                to parse the container.

        Returns:
            The AI generation verdict, or an empty dictionary when the chain is empty
            or carries no matches. C2PA is the only source that never speaks to the
            other two categories because the standard defines no assertions for violent
            or explicit content.
        """
        result = detect_ai_generation(read_manifest_chain(data, media_type))
        return {AI_GENERATED: result} if result.matches else {}
