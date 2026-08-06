"""
The catalogue of C2PA provenance signals.

These signals carry the highest confidences of any layer, and deservedly so. A C2PA
manifest is cryptographically signed, meaning that unlike a standard EXIF tag, it cannot
be edited without breaking the signature. What it states about how an asset was made is
a verifiable claim that its signer staked a certificate on.
"""

from __future__ import annotations

from guard_local.detection import AI_GENERATED, Signal

__all__ = ["AI_SIGNALS"]


class AI_SIGNALS:  # noqa: N801
    """
    The provenance signals ordered roughly by how conclusive they are.

    This class is used purely as a namespace. It mirrors the object literal that the
    browser extension keys into using these exact same names.
    """

    generative_info_assertion = Signal(
        id="c2pa.ai.generative_info",
        category=AI_GENERATED,
        label="c2pa.ai.generative_info",
        description="Explicit AI assertion (since C2PA 2.0)",
        confidence=95,
    )
    ai_generated_action = Signal(
        id="c2pa.ai_generated",
        category=AI_GENERATED,
        label="c2pa.ai_generated action",
        description="Explicit action marker for AI generation",
        confidence=95,
    )
    ai_digital_source_type = Signal(
        id="digitalSourceType.ai",
        category=AI_GENERATED,
        label="AI-indicating digitalSourceType",
        description="IPTC digitalSourceType names an algorithmic/AI source",
        confidence=90,
    )
    software_agent_vendor = Signal(
        id="softwareAgent.vendor",
        category=AI_GENERATED,
        label="AI tool name in softwareAgent",
        description="DALL-E, Firefly, Midjourney, Stable Diffusion, ...",
        confidence=85,
    )
    action_description_vendor = Signal(
        id="action.description.vendor",
        category=AI_GENERATED,
        label="AI tool name in action description",
        description=(
            "Free-text action description names a known AI vendor (e.g. Gemini, "
            "SynthID)"
        ),
        confidence=80,
    )
    generative_ai_metadata_marker = Signal(
        id="generativeAI",
        category=AI_GENERATED,
        label="generativeAI marker in metadata",
        description="Adobe-specific metadata field",
        confidence=85,
    )
    claim_generator_vendor = Signal(
        id="claim_generator.vendor",
        category=AI_GENERATED,
        label="AI tool in claim generator",
        description="Generator field of the signed claim",
        confidence=80,
    )
    training_mining_assertion = Signal(
        id="c2pa.ai.training_mining",
        category=AI_GENERATED,
        label="c2pa.ai.training_mining",
        description="Indirect - hints at an AI training/data-mining context",
        confidence=40,
    )
