"""
Provides one detector per metadata standard.

Each detector is a pure function that takes the parsed segments and returns the matches
it found. Each reads only its own specific segment. This isolation allows a detector to
be tested with a hand-written dictionary without requiring an actual image.
"""

from __future__ import annotations

from .content_scan import scan_content_terms
from .exif import detect_exif_signals
from .icc import detect_icc_signals
from .ihdr import detect_ihdr_signals
from .iptc import detect_iptc_signals
from .jfif import detect_jfif_signals
from .png_text import detect_png_text_signals
from .xmp import detect_xmp_signals

__all__ = [
    "detect_exif_signals",
    "detect_icc_signals",
    "detect_ihdr_signals",
    "detect_iptc_signals",
    "detect_jfif_signals",
    "detect_png_text_signals",
    "detect_xmp_signals",
    "scan_content_terms",
]
