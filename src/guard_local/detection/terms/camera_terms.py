"""
Camera traces that a physical camera leaves behind in media metadata.

Finding two of these traces matching at once is treated as evidence of a real capture.
The make and model list is matched against field values. The other three lists are
matched against field names because the mere presence of an aperture reading matters
more than the specific number it records.
"""

from __future__ import annotations

from typing import Tuple

__all__ = [
    "CAMERA_MAKE_MODEL_TERMS",
    "CAPTURE_SETTING_FIELD_NAMES",
    "GPS_FIELD_NAMES",
    "LENS_FIELD_NAMES",
]

#: Camera and smartphone manufacturers, matched against Make/Model values.
CAMERA_MAKE_MODEL_TERMS: Tuple[str, ...] = (
    "apple",
    "iphone",
    "canon",
    "nikon",
    "sony",
    "fujifilm",
    "fuji",
    "panasonic",
    "leica",
    "olympus",
    "om system",
    "pentax",
    "ricoh",
    "hasselblad",
    "samsung",
    "google",
    "pixel",
    "xiaomi",
    "huawei",
    "honor",
    "oppo",
    "vivo",
    "dji",
    "gopro",
)

#: Field names describing physical lens properties.
LENS_FIELD_NAMES: Tuple[str, ...] = (
    "lensmodel",
    "lensmake",
    "lens",
    "focallength",
)

#: Field names describing exposure mechanics.
CAPTURE_SETTING_FIELD_NAMES: Tuple[str, ...] = (
    "exposuretime",
    "fnumber",
    "iso",
    "isospeedratings",
    "focallength",
    "flash",
    "whitebalance",
    "meteringmode",
)

#: Field names denoting real-world coordinates.
GPS_FIELD_NAMES: Tuple[str, ...] = (
    "gps",
    "latitude",
    "longitude",
    "gpslatitude",
    "gpslongitude",
)
