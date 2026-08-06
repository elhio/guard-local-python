"""
Reads signed provenance from content credentials.

C2PA provides the strongest evidence this engine can evaluate. Because a manifest is
cryptographically signed, it cannot be quietly edited like a standard EXIF tag. When a
generator records a `c2pa.ai_generated` action, it makes that claim under a verifiable
certificate. This layer only evaluates AI generation because the C2PA standard does not
define metrics for violent or explicit content.

Importing this package does not load the `c2pa` runtime into memory. The reader pulls it
in only upon its first use to keep the initialization fast.
"""

from __future__ import annotations

from .detect import detect_ai_generation
from .detectors import AI_SIGNAL_DETECTORS
from .digital_source_types import AI_DIGITAL_SOURCE_TYPES, is_ai_digital_source_type
from .manifest_actions import Manifest, get_actions, get_software_agent_name
from .manifest_store import get_manifest_chain
from .metadata_markers import contains_generative_ai_marker
from .reader import read_manifest_chain
from .signals import AI_SIGNALS
from .source import C2paSource
from .vendors import matches_known_ai_vendor

__all__ = [
    "AI_DIGITAL_SOURCE_TYPES",
    "AI_SIGNALS",
    "AI_SIGNAL_DETECTORS",
    "C2paSource",
    "Manifest",
    "contains_generative_ai_marker",
    "detect_ai_generation",
    "get_actions",
    "get_manifest_chain",
    "get_software_agent_name",
    "is_ai_digital_source_type",
    "matches_known_ai_vendor",
    "read_manifest_chain",
]
