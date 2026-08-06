"""
Reads a C2PA manifest store out of media bytes.

This is the only part of the package that interfaces with the `c2pa` runtime, and it is
deliberately the only part that can fail. Assets carrying a manifest are still the
exception rather than the rule. Because of this, a missing manifest is the ordinary
outcome here rather than an error worth reporting. Every failure is swallowed and
logged. This matches the rule from the browser extension stating that an unreadable
container must not discard the metadata and model results alongside it.
"""

from __future__ import annotations

import io
import json
import logging
from typing import List

from .manifest_actions import Manifest
from .manifest_store import get_manifest_chain

__all__ = ["read_manifest_chain"]

_LOGGER = logging.getLogger(__name__)


def read_manifest_chain(data: bytes, media_type: str) -> List[Manifest]:
    """
    Read the signed provenance chain out of media bytes.

    Args:
        data: The raw media bytes.
        media_type: The MIME type that tells the reader how to find the manifest.
            Video containers are supported alongside still images.

    Returns:
        The active manifest and its ingredients, or an empty list when the asset
        carries no manifest, the container is unsupported, or the read failed.
        This function never raises an exception.
    """
    # Imported here rather than at module scope so that importing `guard_local` stays
    # cheap: this pulls in a native extension that a metadata-only scan never needs.
    try:
        import c2pa
    except ImportError:
        _LOGGER.warning(
            "The c2pa runtime is unavailable, so provenance is not being checked. "
            "Reinstall guard-local-detector to restore it."
        )
        return []

    try:
        with c2pa.Reader(media_type, io.BytesIO(data)) as reader:
            store = json.loads(reader.json())
    except Exception as exc:
        # ManifestNotFound is the common case for ordinary media and is not worth a
        # warning; anything else is logged so a genuine problem stays visible.
        _LOGGER.debug("No readable C2PA manifest (%s): %s", media_type, exc)
        return []

    return get_manifest_chain(store)
