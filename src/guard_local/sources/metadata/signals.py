"""
The catalogue of metadata heuristics and the weight each one carries.

Every entry names what is being looked for in the category field, where it is looked for
in the standard field, which parameters are inspected, and how much confidence it
carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from guard_local.detection import AI_GENERATED, EXPLICIT, VIOLENT, Signal

__all__ = ["METADATA_SIGNALS", "MetadataSignal"]


@dataclass(frozen=True)
class MetadataSignal(Signal):
    """
    A signal read out of an embedded metadata segment.

    Attributes:
        standard: The segment it is read from.
        parameters: The exact field names inspected. This is carried for documentation
            so the catalogue answers what it looks at without requiring a read of the
            detector itself.
    """

    standard: str = ""
    parameters: Tuple[str, ...] = ()


class METADATA_SIGNALS:  # noqa: N801
    """
    The signals named so detectors reference them rather than repeating text.

    A class is used purely as a namespace. This mirrors the object literal that the
    browser extension keys into by the same names.
    """

    exif_software_vendor = MetadataSignal(
        id="exif.software.vendor",
        category=AI_GENERATED,
        label="AI tool name in EXIF Software field",
        description="The EXIF Software tag names a known AI image generator",
        confidence=85,
        standard="exif",
        parameters=("Software",),
    )
    exif_generation_parameters = MetadataSignal(
        id="exif.source.terms",
        category=AI_GENERATED,
        label="Generation parameters in EXIF comment fields",
        description=(
            "EXIF comment/description fields contain diffusion-model generation "
            "parameters (prompt, seed, sampler, ...)"
        ),
        confidence=90,
        standard="exif",
        parameters=("UserComment", "ImageDescription", "XPComment"),
    )
    exif_typical_ai_dimension = MetadataSignal(
        id="exif.dimension.typical",
        category=AI_GENERATED,
        label="Typical AI image dimensions",
        description=(
            "Image dimensions match a size diffusion models commonly default to "
            "(weak on its own)"
        ),
        confidence=20,
        standard="exif",
        parameters=("ExifImageWidth", "ExifImageHeight", "ImageWidth", "ImageHeight"),
    )
    exif_camera_capture = MetadataSignal(
        id="exif.camera.capture",
        category=AI_GENERATED,
        label="Real camera capture evidence",
        description=(
            "Make/model, lens and/or capture settings are present - this is most "
            "likely a real photo, not AI generated"
        ),
        confidence=5,
        kind="authentic",
        standard="exif",
        parameters=(
            "Make",
            "Model",
            "LensModel",
            "LensMake",
            "FocalLength",
            "ExposureTime",
            "FNumber",
            "ISO",
            "Flash",
            "WhiteBalance",
            "MeteringMode",
        ),
    )
    exif_gps = MetadataSignal(
        id="exif.gps.present",
        category=AI_GENERATED,
        label="GPS location data present",
        description="Real-world GPS coordinates are rarely embedded by AI generators",
        confidence=5,
        kind="authentic",
        standard="exif",
        parameters=("GPSLatitude", "GPSLongitude", "GPSAltitude"),
    )
    xmp_generator_vendor = MetadataSignal(
        id="xmp.creatorTool.vendor",
        category=AI_GENERATED,
        label="AI tool name in XMP CreatorTool/History",
        description=(
            "XMP CreatorTool, History, or softwareAgent names a known AI image "
            "generator"
        ),
        confidence=90,
        standard="xmp",
        parameters=(
            "CreatorTool",
            "History",
            "DerivedFrom",
            "DocumentID",
            "InstanceID",
        ),
    )
    xmp_source_terms = MetadataSignal(
        id="xmp.source.terms",
        category=AI_GENERATED,
        label="Generation parameters or AI source terms in XMP",
        description=(
            "XMP fields contain generation parameters (prompt, seed, sampler, ...) or "
            'explicit "AI generated" wording'
        ),
        confidence=90,
        standard="xmp",
        parameters=("dc:description", "DigitalSourceType", "Source"),
    )
    iptc_generator_vendor = MetadataSignal(
        id="iptc.fields.vendor",
        category=AI_GENERATED,
        label="AI tool name in IPTC caption/credit fields",
        description=(
            "IPTC Caption, Credit, Source or Special Instructions name a known AI "
            "image generator"
        ),
        confidence=80,
        standard="iptc",
        parameters=("Caption", "Credit", "Source", "SpecialInstructions", "Keywords"),
    )
    iptc_camera_capture = MetadataSignal(
        id="iptc.source.camera",
        category=AI_GENERATED,
        label="Camera capture wording in IPTC",
        description="IPTC Source/Credit explicitly describes a camera/digital capture",
        confidence=10,
        kind="authentic",
        standard="iptc",
        parameters=("Source", "Credit"),
    )
    icc_vendor_profile = MetadataSignal(
        id="icc.profile.vendor",
        category=AI_GENERATED,
        label="AI tool name in ICC profile description",
        description=(
            "The embedded ICC color profile description names a known AI image "
            "generator (rare, but seen in some exports)"
        ),
        confidence=60,
        standard="icc",
        parameters=("ProfileDescription", "Copyright"),
    )
    jfif_present = MetadataSignal(
        id="jfif.present",
        category=AI_GENERATED,
        label="JFIF marker present",
        description=(
            "A JFIF (APP0) segment is common in standard JPEG encoders; on its own "
            "this says little either way"
        ),
        confidence=10,
        standard="jfif",
        parameters=("JFIFVersion", "ResolutionUnit", "XResolution", "YResolution"),
    )
    ihdr_typical_ai_dimension = MetadataSignal(
        id="ihdr.dimension.typical",
        category=AI_GENERATED,
        label="Typical AI image dimensions (PNG)",
        description=(
            "PNG IHDR width/height match a size diffusion models commonly default to "
            "(weak on its own)"
        ),
        confidence=25,
        standard="ihdr",
        parameters=("ImageWidth", "ImageHeight", "BitDepth", "ColorType"),
    )
    iptc_violent_content = MetadataSignal(
        id="iptc.content.violent",
        category=VIOLENT,
        label="Violent content wording in IPTC",
        description=(
            "IPTC keywords/caption/category fields describe violent or graphic content"
        ),
        confidence=70,
        kind="violence",
        standard="iptc",
        parameters=(
            "Keywords",
            "Caption",
            "Category",
            "SupplementalCategories",
            "Headline",
        ),
    )
    iptc_explicit_content = MetadataSignal(
        id="iptc.content.explicit",
        category=EXPLICIT,
        label="Explicit content wording in IPTC",
        description=(
            "IPTC keywords/caption/category fields describe sexually explicit or adult "
            "content"
        ),
        confidence=80,
        kind="explicit",
        standard="iptc",
        parameters=(
            "Keywords",
            "Caption",
            "Category",
            "SupplementalCategories",
            "Headline",
        ),
    )
    xmp_violent_content = MetadataSignal(
        id="xmp.content.violent",
        category=VIOLENT,
        label="Violent content wording in XMP",
        description=(
            "XMP subject/description/title fields describe violent or graphic content"
        ),
        confidence=70,
        kind="violence",
        standard="xmp",
        parameters=("dc:subject", "dc:description", "dc:title", "Rating"),
    )
    xmp_explicit_content = MetadataSignal(
        id="xmp.content.explicit",
        category=EXPLICIT,
        label="Explicit content wording in XMP",
        description=(
            "XMP subject/description/title fields describe sexually explicit or adult "
            "content"
        ),
        confidence=80,
        kind="explicit",
        standard="xmp",
        parameters=("dc:subject", "dc:description", "dc:title", "Rating"),
    )
    exif_violent_content = MetadataSignal(
        id="exif.content.violent",
        category=VIOLENT,
        label="Violent content wording in EXIF description",
        description=(
            "EXIF description/comment/keyword fields describe violent or graphic "
            "content"
        ),
        confidence=70,
        kind="violence",
        standard="exif",
        parameters=(
            "ImageDescription",
            "UserComment",
            "XPComment",
            "XPKeywords",
            "XPSubject",
        ),
    )
    exif_explicit_content = MetadataSignal(
        id="exif.content.explicit",
        category=EXPLICIT,
        label="Explicit content wording in EXIF description",
        description=(
            "EXIF description/comment/keyword fields describe sexually explicit or "
            "adult content"
        ),
        confidence=80,
        kind="explicit",
        standard="exif",
        parameters=(
            "ImageDescription",
            "UserComment",
            "XPComment",
            "XPKeywords",
            "XPSubject",
        ),
    )
    png_text_generation_parameters = MetadataSignal(
        id="pngtext.source.terms",
        category=AI_GENERATED,
        label="Generation parameters in PNG text chunks",
        description=(
            "A PNG text chunk contains diffusion-model generation parameters (prompt, "
            "seed, sampler, ...), as written by Stable Diffusion, ComfyUI and "
            "Automatic1111"
        ),
        confidence=90,
        standard="pngtext",
        parameters=("parameters", "prompt", "workflow", "Comment", "Description"),
    )
    png_text_vendor = MetadataSignal(
        id="pngtext.software.vendor",
        category=AI_GENERATED,
        label="AI tool name in PNG text chunks",
        description="A PNG text chunk names a known AI image generator",
        confidence=85,
        standard="pngtext",
        parameters=("Software", "Source", "Author", "parameters"),
    )
    png_text_violent_content = MetadataSignal(
        id="pngtext.content.violent",
        category=VIOLENT,
        label="Violent content wording in PNG text chunks",
        description="A PNG text chunk describes violent or graphic content",
        confidence=70,
        kind="violence",
        standard="pngtext",
        parameters=("Description", "Comment", "Title", "parameters"),
    )
    png_text_explicit_content = MetadataSignal(
        id="pngtext.content.explicit",
        category=EXPLICIT,
        label="Explicit content wording in PNG text chunks",
        description="A PNG text chunk describes sexually explicit or adult content",
        confidence=80,
        kind="explicit",
        standard="pngtext",
        parameters=("Description", "Comment", "Title", "parameters"),
    )
