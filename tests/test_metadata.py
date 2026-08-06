"""
Reads what a file says about itself.

The detectors are driven with images built here rather than with committed fixtures, so
every test states exactly which tag it is about. The extraction tests double as a check
on Pillow: each segment is read from a different corner of its API, and a Pillow release
that moves one would otherwise silently reduce this layer to finding nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from guard_local.detection import AI_GENERATED, EXPLICIT, VIOLENT
from guard_local.sources.metadata import (
    collect_camera_evidence,
    detect_metadata_signals,
    extract_image_metadata,
    find_matching_term,
    flatten_to_searchable_text,
    has_any_field,
    has_strong_camera_evidence,
)

from .conftest import jpeg_bytes, png_bytes, png_with_text, tagged_jpeg, xmp_jpeg


def signals(data: bytes, media_type: str, category: str) -> dict:
    """Score one image and return that category's signal ids mapped to confidences."""
    result = detect_metadata_signals(extract_image_metadata(data, media_type))
    found = result.get(category)
    return {m.id: m.confidence for m in found.matches} if found else {}


class TestTextMatch:
    def test_keys_are_searched_alongside_values(self) -> None:
        """A field *named* GenerativeAI is as telling as one whose value says so."""
        text = flatten_to_searchable_text({"GenerativeAI": True})

        assert "generativeai" in text.lower()

    def test_nested_values_are_reached(self) -> None:
        text = flatten_to_searchable_text({"a": {"b": ["c", {"d": "midjourney"}]}})

        assert "midjourney" in text

    def test_the_walk_stops_at_the_depth_cap(self) -> None:
        """
        Ensure the flattening depth is capped.

        Metadata is attacker-supplied and arbitrarily nestable, so the walk must not
        follow it indefinitely.
        """
        deep: Any = "buried"
        for _ in range(12):
            deep = {"next": deep}

        assert "buried" not in flatten_to_searchable_text(deep)

    def test_bytes_are_decoded_rather_than_stringified(self) -> None:
        """XMP arrives raw; repr'ing it would still match, but on the wrong text."""
        assert "midjourney" in flatten_to_searchable_text(b"Midjourney").lower()

    def test_a_malformed_byte_does_not_lose_the_packet(self) -> None:
        assert "firefly" in flatten_to_searchable_text(b"\xff\xfefirefly").lower()

    def test_matching_is_case_insensitive(self) -> None:
        assert find_matching_term({"Software": "MIDJOURNEY"}, ["midjourney"])

    def test_the_first_listed_term_wins(self) -> None:
        found = find_matching_term({"x": "sora and flux"}, ["flux", "sora"])

        assert found == "flux"

    def test_nothing_matches_an_empty_segment(self) -> None:
        assert find_matching_term(None, ["midjourney"]) is None
        assert find_matching_term({}, ["midjourney"]) is None

    def test_field_lookup_ignores_case_and_only_scans_the_top_level(self) -> None:
        assert has_any_field({"FNumber": 1.8}, ["fnumber"]) is True
        assert has_any_field({"nested": {"FNumber": 1.8}}, ["fnumber"]) is False

    def test_field_lookup_does_not_care_what_the_value_is(self) -> None:
        """Presence of an aperture reading is the signal, not its value."""
        assert has_any_field({"FNumber": 0}, ["fnumber"]) is True


class TestExtraction:
    def test_exif_tags_come_back_by_name(self) -> None:
        metadata = extract_image_metadata(
            tagged_jpeg(**{"0x0131": "Midjourney"}), "image/jpeg"
        )

        assert metadata["ifd0"]["Software"] == "Midjourney"

    def test_the_jfif_segment_is_detected(self) -> None:
        assert "jfif" in extract_image_metadata(jpeg_bytes(), "image/jpeg")

    def test_png_headers_are_read_from_the_bytes(self) -> None:
        """
        Ensure the PNG header is parsed directly rather than derived from Pillow.

        Bit depth is not exposed on a decoded image, so deriving it from the mode
        would be a guess; the header chunk states it.
        """
        metadata = extract_image_metadata(png_bytes(1024, 1024), "image/png")

        assert metadata["ihdr"] == {
            "ImageWidth": 1024,
            "ImageHeight": 1024,
            "BitDepth": 8,
            "ColorType": 2,
        }

    def test_png_text_chunks_are_read(self) -> None:
        """The browser extension's parser reads only the header and misses these."""
        metadata = extract_image_metadata(
            png_with_text(parameters="Steps: 20"), "image/png"
        )

        assert metadata["png_text"]["parameters"] == "Steps: 20"

    def test_xmp_is_kept_as_raw_text(self) -> None:
        data = xmp_jpeg("<x:xmpmeta>hi</x:xmpmeta>")

        metadata = extract_image_metadata(data, "image/jpeg")

        assert "<x:xmpmeta>" in metadata["xmp"]

    def test_absent_segments_are_omitted_rather_than_empty(self) -> None:
        metadata = extract_image_metadata(png_bytes(), "image/png")

        assert "exif" not in metadata
        assert "iptc" not in metadata

    def test_video_is_not_opened_at_all(self) -> None:
        """A container's provenance lives in its manifest, not in EXIF."""
        assert extract_image_metadata(jpeg_bytes(), "video/mp4") == {}

    def test_corrupt_bytes_are_treated_as_carrying_no_metadata(self) -> None:
        """
        Ensure an unreadable file yields nothing instead of raising.

        A metadata failure must never sink a scan whose model and provenance layers
        would otherwise have succeeded.
        """
        assert extract_image_metadata(b"not an image", "image/png") == {}


class TestExifSignals:
    def test_a_generator_in_the_software_tag_fires(self) -> None:
        found = signals(
            tagged_jpeg(**{"0x0131": "Midjourney v6"}), "image/jpeg", AI_GENERATED
        )

        assert found["exif.software.vendor"] == 85

    def test_generation_parameters_in_a_comment_fire(self) -> None:
        found = signals(
            tagged_jpeg(**{"0x9286": "Steps: 20, Sampler: Euler a, Seed: 12"}),
            "image/jpeg",
            AI_GENERATED,
        )

        assert found["exif.source.terms"] == 90

    def test_a_generator_elsewhere_does_not_fire_the_software_signal(self) -> None:
        """
        Ensure each signal only reads the fields it declares.

        Otherwise its evidence string would claim a match came from the Software tag
        when it came from a caption.
        """
        found = signals(
            tagged_jpeg(**{"0x9286": "Midjourney"}), "image/jpeg", AI_GENERATED
        )

        assert "exif.software.vendor" not in found

    def test_two_camera_traces_are_needed_before_capture_evidence_fires(self) -> None:
        one = signals(tagged_jpeg(**{"0x010F": "Apple"}), "image/jpeg", AI_GENERATED)
        two = signals(
            tagged_jpeg(**{"0x010F": "Apple", "0x829D": 1.8}),
            "image/jpeg",
            AI_GENERATED,
        )

        assert "exif.camera.capture" not in one
        assert two["exif.camera.capture"] == 5

    def test_every_jpeg_carries_the_jfif_marker(self) -> None:
        """
        Ensure the weak JFIF floor is reported, as the browser extension reports it.

        This is a deliberately weak signal at confidence 10. It is asserted here so
        that the floor it puts under every JPEG's AI score is a visible choice rather
        than a surprise.
        """
        assert signals(jpeg_bytes(), "image/jpeg", AI_GENERATED) == {"jfif.present": 10}


class TestContentSignals:
    def test_explicit_wording_in_a_comment_fires(self) -> None:
        found = signals(
            tagged_jpeg(**{"0x9286": "tagged NSFW by the uploader"}),
            "image/jpeg",
            EXPLICIT,
        )

        assert found["exif.content.explicit"] == 80

    def test_violent_wording_in_a_comment_fires(self) -> None:
        found = signals(
            tagged_jpeg(**{"0x010E": "depicts graphic violence"}),
            "image/jpeg",
            VIOLENT,
        )

        assert found["exif.content.violent"] == 70

    def test_ambiguous_stems_are_deliberately_not_matched(self) -> None:
        """
        Ensure short ambiguous stems cannot fire a content signal.

        Terms are matched by substring, so `gore` would match "category" and turn
        every categorised photo into a violence hit.
        """
        found = signals(
            tagged_jpeg(**{"0x010E": "category: landscape"}), "image/jpeg", VIOLENT
        )

        assert found == {}


class TestXmpSignals:
    def test_a_generator_in_the_packet_fires(self) -> None:
        packet = (
            '<x:xmpmeta><rdf:Description xmp:CreatorTool="Adobe Firefly"/></x:xmpmeta>'
        )

        found = signals(xmp_jpeg(packet), "image/jpeg", AI_GENERATED)

        assert found["xmp.creatorTool.vendor"] == 90

    def test_the_packet_is_searched_without_being_parsed(self) -> None:
        """
        Ensure XMP is matched as raw text.

        A namespace this code has never seen is still scanned, and no XML parser (nor
        the dependency one would need) is involved.
        """
        packet = "<x:xmpmeta><madeUpNs:f>ai generated</madeUpNs:f></x:xmpmeta>"

        found = signals(xmp_jpeg(packet), "image/jpeg", AI_GENERATED)

        assert found["xmp.source.terms"] == 90


class TestPngTextSignals:
    def test_diffusion_parameters_are_caught(self) -> None:
        """
        Ensure a PNG's generation parameters are read.

        This closes the browser extension's largest blind spot: Stable Diffusion,
        ComfyUI and Automatic1111 write the whole generation here and nothing at all
        to EXIF or XMP, so without this the file looks like an ordinary picture.
        """
        data = png_with_text(
            parameters="a cat\nNegative prompt: blurry\nSteps: 20, Sampler: Euler a"
        )

        found = signals(data, "image/png", AI_GENERATED)

        assert found["pngtext.source.terms"] == 90

    def test_a_generator_named_in_a_chunk_is_caught(self) -> None:
        found = signals(png_with_text(Software="ComfyUI"), "image/png", AI_GENERATED)

        assert found["pngtext.software.vendor"] == 85

    def test_content_wording_in_a_chunk_is_caught(self) -> None:
        found = signals(
            png_with_text(parameters="nsfw, explicit content"), "image/png", EXPLICIT
        )

        assert found["pngtext.content.explicit"] == 80

    def test_a_plain_png_fires_nothing(self) -> None:
        assert signals(png_bytes(), "image/png", AI_GENERATED) == {}


class TestIhdrSignals:
    def test_a_generator_default_size_fires(self) -> None:
        found = signals(png_bytes(1024, 1024), "image/png", AI_GENERATED)

        assert found["ihdr.dimension.typical"] == 25

    def test_an_ordinary_size_does_not(self) -> None:
        assert signals(png_bytes(640, 480), "image/png", AI_GENERATED) == {}


class TestWebpCoverage:
    def test_webp_metadata_is_read(self) -> None:
        """
        Ensure WebP metadata is read at all.

        The browser extension's parser rejects the container outright and treats every
        WebP as carrying nothing, which is the second gap this layer closes.
        """
        from .conftest import encode, gradient

        data = encode(
            gradient(), "WEBP", xmp=b'<x:xmpmeta>CreatorTool="Midjourney"</x:xmpmeta>'
        )

        assert signals(data, "image/webp", AI_GENERATED)["xmp.creatorTool.vendor"] == 90


class TestCameraEvidence:
    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            ((True, False, False, False), False),
            ((True, True, False, False), True),
            ((False, False, True, True), True),
            ((True, True, True, True), True),
            ((False, False, False, False), False),
        ],
    )
    def test_two_independent_traces_are_required(
        self, flags: tuple, expected: bool
    ) -> None:
        """
        Ensure a single trace is never enough.

        Any one field is guessable and forgeable; two independent ones are what turn
        it into evidence.
        """
        from guard_local.sources.metadata import CameraEvidence

        assert has_strong_camera_evidence(CameraEvidence(*flags)) is expected

    def test_the_gps_segment_counts_even_when_its_fields_are_unnamed(self) -> None:
        evidence = collect_camera_evidence({"gps": {0x0002: (51, 30, 0)}})

        assert evidence.has_gps is True

    def test_ifd0_and_the_exif_directory_are_read_together(self) -> None:
        """Which of the two holds a given tag varies by whoever wrote the file."""
        evidence = collect_camera_evidence(
            {"ifd0": {"Make": "Canon"}, "exif": {"FNumber": 1.8}}
        )

        assert evidence.has_make_model and evidence.has_capture_settings
