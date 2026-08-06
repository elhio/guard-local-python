"""
Wording that betrays how an image was made.

The `AI_SOURCE_TERMS` list contains the leftovers of a generation run. These include the
IPTC source vocabulary that a generator writes, as well as the diffusion parameters like
prompt, seed, and sampler that tools dump verbatim into comment fields. The
`CAMERA_SOURCE_TERMS` list represents the opposite claim, which is typically made in
IPTC metadata by wire services and camera firmware.
"""

from __future__ import annotations

from typing import Tuple

__all__ = ["AI_SOURCE_TERMS", "CAMERA_SOURCE_TERMS"]

AI_SOURCE_TERMS: Tuple[str, ...] = (
    "trainedalgorithmicmedia",
    "compositewithtrainedalgorithmicmedia",
    "algorithmicmedia",
    "generative ai",
    "ai generated",
    "ai-generated",
    "synthetic media",
    "text-to-image",
    "txt2img",
    "img2img",
    "prompt",
    "negative prompt",
    "seed",
    "sampler",
    "cfg scale",
    "denoising strength",
)

CAMERA_SOURCE_TERMS: Tuple[str, ...] = (
    "digitalcapture",
    "digital capture",
    "original digital capture",
    "camera capture",
    "captured by camera",
)
