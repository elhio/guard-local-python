"""
Provenance detectors where each identifies a specific admission of AI involvement.

Each detector takes a single manifest and returns at most one match. This allows the
caller to decide how to traverse the manifest chain. The detectors read the manifest
defensively because the reader returns exactly what the signer wrote, meaning a missing
field is treated as a normal condition rather than an error.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from guard_local.detection import SignalMatch

from .digital_source_types import is_ai_digital_source_type
from .manifest_actions import Manifest, get_actions, get_software_agent_name
from .metadata_markers import contains_generative_ai_marker
from .signals import AI_SIGNALS
from .vendors import matches_known_ai_vendor

__all__ = [
    "AI_SIGNAL_DETECTORS",
    "detect_ai_action_description",
    "detect_ai_claim_generator",
    "detect_ai_digital_source_type",
    "detect_ai_generated_action",
    "detect_ai_software_agent",
    "detect_generative_ai_metadata_marker",
    "detect_generative_info_assertion",
    "detect_training_or_mining_assertion",
]


def _manifest_ref(manifest: Manifest) -> str:
    """
    Name a manifest to be used in an evidence string.

    Args:
        manifest: The manifest being quoted.

    Returns:
        The label of the manifest, or a fallback stand-in when it has none.
    """
    return manifest.get("label") or "(unlabeled manifest)"


def _find_assertion(manifest: Manifest, prefix: str) -> Optional[Dict[str, Any]]:
    """
    Find the first assertion whose label starts with a given prefix.

    Args:
        manifest: The manifest to search.
        prefix: The assertion label prefix, which also matches versioned suffixes.

    Returns:
        The matching assertion if found, otherwise `None`.
    """
    for assertion in manifest.get("assertions") or []:
        if str(assertion.get("label", "")).startswith(prefix):
            found: Dict[str, Any] = assertion
            return found
    return None


def detect_generative_info_assertion(manifest: Manifest) -> Optional[SignalMatch]:
    """
    Look for the explicit generative info assertion introduced in C2PA 2.0.

    Args:
        manifest: The manifest to evaluate.

    Returns:
        A match when the assertion is present, otherwise `None`.
    """
    assertion = _find_assertion(manifest, "c2pa.ai.generative_info")
    if assertion is None:
        return None
    return SignalMatch.of(
        AI_SIGNALS.generative_info_assertion,
        f'assertion "{assertion.get("label")}" present in {_manifest_ref(manifest)}',
        "c2pa",
    )


def detect_ai_generated_action(manifest: Manifest) -> Optional[SignalMatch]:
    """
    Look for an action explicitly tagged as AI generation.

    Args:
        manifest: The manifest to evaluate.

    Returns:
        A match when such an action is recorded, otherwise `None`.
    """
    for action in get_actions(manifest):
        name = str(action.get("action", ""))
        if name.endswith("ai_generated"):
            return SignalMatch.of(
                AI_SIGNALS.ai_generated_action,
                f'action "{name}" in {_manifest_ref(manifest)}',
                "c2pa",
            )
    return None


def detect_ai_digital_source_type(manifest: Manifest) -> Optional[SignalMatch]:
    """
    Look for an action declaring an algorithmic digital source.

    Args:
        manifest: The manifest to evaluate.

    Returns:
        A match when an action names an algorithmic source type, otherwise `None`.
    """
    for action in get_actions(manifest):
        source_type = action.get("digitalSourceType")
        if is_ai_digital_source_type(source_type):
            return SignalMatch.of(
                AI_SIGNALS.ai_digital_source_type,
                f'digitalSourceType "{source_type}" in {_manifest_ref(manifest)}',
                "c2pa",
            )
    return None


def detect_ai_software_agent(manifest: Manifest) -> Optional[SignalMatch]:
    """
    Look for a known generator named as the agent that performed an action.

    Args:
        manifest: The manifest to evaluate.

    Returns:
        A match when an agent name matches a known vendor, otherwise `None`.
    """
    for action in get_actions(manifest):
        name = get_software_agent_name(action.get("softwareAgent"))
        vendor = matches_known_ai_vendor(name)
        if vendor:
            return SignalMatch.of(
                AI_SIGNALS.software_agent_vendor,
                f'softwareAgent "{name}" matches known AI vendor "{vendor}" in '
                f"{_manifest_ref(manifest)}",
                "c2pa",
            )
    return None


def detect_ai_action_description(manifest: Manifest) -> Optional[SignalMatch]:
    """
    Look for a known generator named in the free text description of an action.

    Args:
        manifest: The manifest to evaluate.

    Returns:
        A match when a description names a known vendor, otherwise `None`.
    """
    for action in get_actions(manifest):
        description = action.get("description")
        vendor = matches_known_ai_vendor(description)
        if vendor:
            return SignalMatch.of(
                AI_SIGNALS.action_description_vendor,
                f'action description "{description}" matches known AI vendor '
                f'"{vendor}" in {_manifest_ref(manifest)}',
                "c2pa",
            )
    return None


def detect_generative_ai_metadata_marker(manifest: Manifest) -> Optional[SignalMatch]:
    """
    Look for a generative AI marker buried inside any assertion payload.

    Args:
        manifest: The manifest to evaluate.

    Returns:
        A match when a marker is found, otherwise `None`.
    """
    for assertion in manifest.get("assertions") or []:
        if contains_generative_ai_marker(assertion.get("data")):
            return SignalMatch.of(
                AI_SIGNALS.generative_ai_metadata_marker,
                f'assertion "{assertion.get("label")}" contains a generativeAI '
                f"marker in {_manifest_ref(manifest)}",
                "c2pa",
            )
    return None


def detect_ai_claim_generator(manifest: Manifest) -> Optional[SignalMatch]:
    """
    Look for a known generator named as the signer of the claim.

    Args:
        manifest: The manifest to evaluate.

    Returns:
        A match when the claim generator matches a known vendor, otherwise `None`.
    """
    candidates = [manifest.get("claim_generator")]
    for info in manifest.get("claim_generator_info") or []:
        if isinstance(info, dict):
            candidates.append(info.get("name"))

    for candidate in candidates:
        vendor = matches_known_ai_vendor(candidate)
        if vendor:
            return SignalMatch.of(
                AI_SIGNALS.claim_generator_vendor,
                f'claim generator "{candidate}" matches known AI vendor "{vendor}" '
                f"in {_manifest_ref(manifest)}",
                "c2pa",
            )
    return None


def detect_training_or_mining_assertion(manifest: Manifest) -> Optional[SignalMatch]:
    """
    Look for a training or data-mining assertion.

    Args:
        manifest: The manifest to evaluate.

    Returns:
        A match when the assertion is present, otherwise `None`.
    """
    assertion = _find_assertion(manifest, "c2pa.ai.training_mining")
    if assertion is None:
        return None
    return SignalMatch.of(
        AI_SIGNALS.training_mining_assertion,
        f'assertion "{assertion.get("label")}" present in {_manifest_ref(manifest)}',
        "c2pa",
    )


#: Every detector run against every manifest in a chain.
AI_SIGNAL_DETECTORS: Tuple[Callable[[Manifest], Optional[SignalMatch]], ...] = (
    detect_generative_info_assertion,
    detect_ai_generated_action,
    detect_ai_digital_source_type,
    detect_ai_software_agent,
    detect_ai_action_description,
    detect_generative_ai_metadata_marker,
    detect_ai_claim_generator,
    detect_training_or_mining_assertion,
)
