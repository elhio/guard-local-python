"""
The independent sources of evidence and the unified format they use to report results.

Each source reads the same media bytes and reports what it found per category. They
operate entirely independently of one another. For example, the provenance layer does
not know a vision model exists, and the vision model does not know whether a file
contains an EXIF tag. Weighing these disparate signals against each other is the
exclusive responsibility of the engine.

The sources are ordered by their level of trust. A cryptographically signed manifest
cannot be edited without breaking its signature, making it highly reliable. A metadata
tag can be stripped or forged by anyone, but when present, it is usually a generator
explicitly identifying itself. A vision model only ever offers an educated assessment,
but it serves as the final fallback when a file carries no metadata at all.
"""

from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable

from guard_local.detection import CategoryResult

from .c2pa import C2paSource
from .metadata import MetadataSource
from .model import ModelSource

__all__ = ["C2paSource", "MetadataSource", "ModelSource", "Source", "SourceResult"]

#: What every source returns: a verdict for each category it evaluated. Categories
#: that a source stayed silent on are omitted rather than returned empty.
SourceResult = Dict[str, CategoryResult]


@runtime_checkable
class Source(Protocol):
    """
    An interface for any component the engine can query for an evaluation.

    name: The label this source applies to the matches it produces. This is
            the exact string that appears as the `source` attribute on every
            `SignalMatch` it returns.
    """

    name: str

    def analyze(self, data: bytes, media_type: str) -> SourceResult:
        """
        Evaluate the media and report what was found.

        Args:
            data: The raw media bytes.
            media_type: The MIME type of the media.

        Returns:
            A verdict for each category this source evaluated.
        """
        ...
