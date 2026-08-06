"""
Extracts the embedded metadata segments from image bytes.

Each segment is kept in its own key rather than merged into one flat bag. This isolation
is load-bearing because it allows an EXIF detector to scan only EXIF data. As a result,
a generator name in an IPTC caption cannot fire a signal claiming it came from the EXIF
Software tag, ensuring the evidence string remains accurate.

Everything here operates on a best-effort basis by design. An unreadable or unsupported
container is not treated as an error. It simply means the file has no metadata. This
approach ensures a missing segment does not sink a scan where the vision model and C2PA
layers would otherwise succeed.
"""

from __future__ import annotations

import io
import struct
from typing import Any, Dict, Optional

import pillow_heif
from PIL import ExifTags, Image, ImageFile, IptcImagePlugin

from guard_local.tasks import VIDEO_MEDIA_TYPES

__all__ = ["RawImageMetadata", "extract_image_metadata"]

# HEIC needs the same opener that the decoding path registers. Registering is idempotent,
# so doing it here allows metadata to be read without ever loading the vision model.
pillow_heif.register_heif_opener()

#: A parsed file's segments keyed by the standard they came from. Missing segments are
#: absent rather than empty so a detector can distinguish not present from bare.
RawImageMetadata = Dict[str, Any]

#: The IPTC IIM datasets worth naming from record 2. Anything unnamed is kept under its
#: raw `(record, dataset)` key so no data is silently dropped.
_IPTC_DATASETS = {
    5: "ObjectName",
    15: "Category",
    20: "SupplementalCategories",
    25: "Keywords",
    40: "SpecialInstructions",
    80: "Byline",
    85: "BylineTitle",
    105: "Headline",
    110: "Credit",
    115: "Source",
    116: "CopyrightNotice",
    120: "Caption",
    122: "Writer",
}

#: The eight-byte signature every PNG starts with.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def extract_image_metadata(data: bytes, media_type: str) -> RawImageMetadata:
    """
    Read every metadata segment a still image carries.

    Args:
        data: The raw media bytes.
        media_type: The MIME type of the media. Video types return an empty
            dictionary because a container's provenance lives in its C2PA
            manifest rather than in EXIF.

    Returns:
        The segments found keyed by standard. Returns an empty mapping when
        the bytes carry no metadata, cannot be decoded, or are not a still
        image. This function never raises an exception.
    """
    if media_type in VIDEO_MEDIA_TYPES:
        return {}

    try:
        return _read_segments(data)
    except Exception:
        # Matches the browser extension, which treats an unparseable file as simply
        # having no metadata rather than failing the whole candidate.
        return {}


def _read_segments(data: bytes) -> RawImageMetadata:
    """
    Open the bytes once and collect each segment from the decoded image.

    Args:
        data: The raw media bytes.

    Returns:
        Every segment that turned out to be present.
    """
    metadata: RawImageMetadata = {}

    with Image.open(io.BytesIO(data)) as image:
        image.load()

        _put(metadata, "ifd0", _read_ifd0(image))
        _put(metadata, "exif", _read_sub_ifd(image, ExifTags.IFD.Exif, ExifTags.TAGS))
        _put(
            metadata,
            "gps",
            _read_sub_ifd(image, ExifTags.IFD.GPSInfo, ExifTags.GPSTAGS),
        )
        _put(metadata, "xmp", _read_xmp(image))
        _put(metadata, "iptc", _read_iptc(image))
        _put(metadata, "icc", _read_icc(image))
        _put(metadata, "jfif", _read_jfif(image))
        _put(metadata, "png_text", _read_png_text(image))

    _put(metadata, "ihdr", _read_ihdr(data))
    return metadata


def _put(metadata: RawImageMetadata, key: str, segment: Any) -> None:
    """
    Record a segment and drop it when it turns out to be empty.

    Args:
        metadata: The mapping being built.
        key: The segment name.
        segment: What was read, which may be empty or `None`.
    """
    if segment:
        metadata[key] = segment


def _read_ifd0(image: Image.Image) -> Dict[str, Any]:
    """
    Read the primary image directory which carries make, model, and software.

    Args:
        image: The decoded image.

    Returns:
        The IFD0 tags by name excluding the pointers to the subdirectories that are read
        separately.
    """
    exif = image.getexif()
    if not exif:
        return {}
    return {
        ExifTags.TAGS.get(tag, str(tag)): value
        for tag, value in exif.items()
        if tag not in (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo)
    }


def _read_sub_ifd(
    image: Image.Image, ifd: ExifTags.IFD, names: Dict[int, str]
) -> Dict[str, Any]:
    """
    Read one of the EXIF subdirectories.

    Args:
        image: The decoded image.
        ifd: Which subdirectory to read.
        names: The tag number to tag name mapping for that directory.

    Returns:
        The subdirectory tags by name, or an empty dictionary when absent.
    """
    exif = image.getexif()
    if not exif:
        return {}
    try:
        block = exif.get_ifd(ifd)
    except Exception:
        return {}
    return {names.get(tag, str(tag)): value for tag, value in (block or {}).items()}


def _read_xmp(image: Image.Image) -> Optional[str]:
    """
    Read the raw XMP packet.

    The packet is deliberately left as XML text rather than parsed. Every XMP signal is
    a case-insensitive substring match over flattened text. Parsing would offer no
    benefits and would cost a `defusedxml` dependency that the native Pillow `getxmp`
    function requires.

    Args:
        image: The decoded image.

    Returns:
        The packet as text, or `None` when the file carries no XMP.
    """
    packet = image.info.get("xmp")
    if not packet:
        return None
    if isinstance(packet, (bytes, bytearray)):
        return bytes(packet).decode("utf-8", errors="replace")
    return str(packet)


def _read_iptc(image: ImageFile.ImageFile) -> Dict[Any, Any]:
    """
    Read the IPTC IIM block and name the datasets worth naming.

    Args:
        image: The decoded image. This is typed as the file-backed subclass because that
            is what `Image.open` returns and what the IPTC reader requires.

    Returns:
        The IPTC fields with record 2 datasets given their conventional names. Values
        are decoded to text because every IPTC signal is a text match.
    """
    try:
        info = IptcImagePlugin.getiptcinfo(image)
    except Exception:
        return {}
    if not info:
        return {}

    fields: Dict[Any, Any] = {}
    for key, value in info.items():
        record, dataset = key if isinstance(key, tuple) else (None, None)
        name: Any = _IPTC_DATASETS.get(dataset, key) if record == 2 else key
        fields[name] = _decode_iptc_value(value)
    return fields


def _decode_iptc_value(value: Any) -> Any:
    """
    Turn an IPTC value into text.

    Args:
        value: One field value. Pillow reports this as bytes or as a list of bytes for
            repeatable datasets such as Keywords.

    Returns:
        The value as a string or list of strings. It is left untouched if it is neither.
    """
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_decode_iptc_value(item) for item in value]
    return value


def _read_icc(image: Image.Image) -> Dict[str, str]:
    """
    Read the description and copyright out of the embedded colour profile.

    Args:
        image: The decoded image.

    Returns:
        The text fields of the profile. This falls back to the raw profile bytes decoded
        as text when the colour management module is unavailable because the signal
        built on this is a substring match either way.
    """
    profile = image.info.get("icc_profile")
    if not profile:
        return {}

    try:
        from PIL import ImageCms

        parsed = ImageCms.ImageCmsProfile(io.BytesIO(profile)).profile
        fields = {
            "ProfileDescription": parsed.profile_description or "",
            "Copyright": parsed.copyright or "",
        }
        return {key: value for key, value in fields.items() if value}
    except Exception:
        return {"ProfileDescription": bytes(profile).decode("latin-1", "replace")}


def _read_jfif(image: Image.Image) -> Dict[str, Any]:
    """
    Read the JFIF APP0 segment.

    Args:
        image: The decoded image.

    Returns:
        The JFIF fields, or an empty dictionary when the file has no APP0 segment.
        Presence is the only thing the signal cares about. The values are carried
        forward purely for the evidence string.
    """
    if "jfif" not in image.info:
        return {}

    density = image.info.get("jfif_density") or (None, None)
    fields = {
        "JFIFVersion": image.info.get("jfif_version"),
        "ResolutionUnit": image.info.get("jfif_unit"),
        "XResolution": density[0],
        "YResolution": density[1],
    }
    return {key: value for key, value in fields.items() if value is not None}


def _read_png_text(image: Image.Image) -> Dict[str, str]:
    """
    Read the PNG `tEXt`, `zTXt`, and `iTXt` chunks.

    Args:
        image: The decoded image.

    Returns:
        The text chunks by keyword, or an empty dictionary for a non-PNG file.
    """
    chunks = getattr(image, "text", None)
    return dict(chunks) if chunks else {}


def _read_ihdr(data: bytes) -> Dict[str, int]:
    """
    Read a PNG header chunk straight from the bytes.

    The header is at a fixed offset immediately after the signature. Reading it here is
    exact, whereas deriving it from the Pillow mode would require guessing the bit
    depth.

    Args:
        data: The raw media bytes.

    Returns:
        The declared dimensions, bit depth, and colour type, or an empty dictionary for
        a non-PNG or a file too short to hold a header.
    """
    if not data.startswith(_PNG_SIGNATURE) or len(data) < 26:
        return {}
    width, height, bit_depth, colour_type = struct.unpack(">IIBB", data[16:26])
    return {
        "ImageWidth": width,
        "ImageHeight": height,
        "BitDepth": bit_depth,
        "ColorType": colour_type,
    }
