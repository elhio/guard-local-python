"""
The word and number lists the detectors match against.

These lists are kept separate from the detectors that use them. This separation ensures
that keeping them synchronized with other clients is a simple matter of comparing two
lists rather than deciphering two different implementations.
"""

from __future__ import annotations

from .ai_dimensions import TYPICAL_AI_DIMENSIONS, looks_like_typical_ai_dimension
from .ai_source_terms import AI_SOURCE_TERMS, CAMERA_SOURCE_TERMS
from .ai_vendors import AI_GENERATOR_VENDORS
from .camera_terms import (
    CAMERA_MAKE_MODEL_TERMS,
    CAPTURE_SETTING_FIELD_NAMES,
    GPS_FIELD_NAMES,
    LENS_FIELD_NAMES,
)
from .content_terms import EXPLICIT_TERMS, VIOLENCE_TERMS

__all__ = [
    "AI_GENERATOR_VENDORS",
    "AI_SOURCE_TERMS",
    "CAMERA_MAKE_MODEL_TERMS",
    "CAMERA_SOURCE_TERMS",
    "CAPTURE_SETTING_FIELD_NAMES",
    "EXPLICIT_TERMS",
    "GPS_FIELD_NAMES",
    "LENS_FIELD_NAMES",
    "TYPICAL_AI_DIMENSIONS",
    "VIOLENCE_TERMS",
    "looks_like_typical_ai_dimension",
]
