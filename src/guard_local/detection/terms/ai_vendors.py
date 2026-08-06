"""
AI vendors indicating that an AI generator created or modified the media.

Both the metadata detectors and the C2PA vendor matcher read this list. This ensures the
two layers never disagree about what counts as a generator. Entries are lowercased name
fragments designed for lenient substring matching. This lenient matching is necessary
because the fields they are matched against often carry versioned forms like
`Product/1.0` or `Service API` rather than bare names.
"""

from __future__ import annotations

from typing import Tuple

__all__ = ["AI_GENERATOR_VENDORS"]

AI_GENERATOR_VENDORS: Tuple[str, ...] = (
    "dall-e",
    "dalle",
    "openai",
    "chatgpt",
    "gpt-image",
    "midjourney",
    "stable diffusion",
    "stablediffusion",
    "sdxl",
    "stability ai",
    "stability.ai",
    "flux",
    "black forest labs",
    "comfyui",
    "automatic1111",
    "a1111",
    "invokeai",
    "leonardo",
    "leonardo.ai",
    "leonardo ai",
    "ideogram",
    "firefly",
    "adobe firefly",
    "runway",
    "runwayml",
    "runway ml",
    "sora",
    "imagen",
    "gemini",
    "google generative ai",
    "synthid",
    "veo",
    "recraft",
    "krea",
    "nightcafe",
    "dreamstudio",
    "playground ai",
    "craiyon",
    "bing image creator",
    "copilot designer",
    "grok imagine",
)
