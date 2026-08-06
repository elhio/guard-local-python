"""
The model source: decoding, the transform, the session, and the scores.

The transform is asserted geometrically rather than by eye. It must match the exact
transform the browser extension runs against the same ONNX file. Every step of this
transform, including truncation, centering, edge padding, and channel order, is a
potential place where the two implementations could silently drift apart.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

from guard_local.detection import CATEGORIES, CategoryResult
from guard_local.exceptions import (
    MediaDecodeError,
    ModelLoadError,
    UnsupportedMediaError,
)
from guard_local.sources.model import (
    IMAGE_SIZE,
    ModelSession,
    ModelSource,
    letterbox,
    load_frames,
    to_tensor,
)
from guard_local.sources.model.source import _sigmoid

from .conftest import (
    exif_rotated_jpeg,
    gif_bytes,
    gradient,
    heic_bytes,
    jpeg_bytes,
    png_bytes,
    video_bytes,
    webp_bytes,
)


@pytest.mark.parametrize(
    ("data", "media_type"),
    [
        (png_bytes(), "image/png"),
        (jpeg_bytes(), "image/jpeg"),
        (webp_bytes(), "image/webp"),
        (gif_bytes(), "image/gif"),
        (heic_bytes(), "image/heic"),
    ],
)
def test_every_still_format_decodes_to_one_rgb_frame(
    data: bytes, media_type: str
) -> None:
    frames = load_frames(data, media_type)

    assert len(frames) == 1
    assert frames[0].mode == "RGB"


def test_animated_gif_yields_its_first_frame() -> None:
    """Ensure animated GIFs yield only their first frame, as the extension does."""
    (frame,) = load_frames(gif_bytes(), "image/gif")

    assert frame.getpixel((0, 0)) == (255, 0, 0)


def test_exif_orientation_is_applied() -> None:
    """Ensure EXIF orientation is applied during decoding."""
    (frame,) = load_frames(exif_rotated_jpeg(), "image/jpeg")

    assert frame.size == (32, 64)  # the 64x32 source, stood up by orientation 6


def test_alpha_is_dropped_not_composited() -> None:
    """Ensure the alpha channel is dropped rather than composited onto a background."""
    transparent = Image.new("RGBA", (8, 8), (10, 20, 30, 0))
    buffer = __import__("io").BytesIO()
    transparent.save(buffer, format="PNG")

    (frame,) = load_frames(buffer.getvalue(), "image/png")

    assert frame.getpixel((0, 0)) == (10, 20, 30)


def test_unsupported_media_type_is_rejected_by_name() -> None:
    with pytest.raises(UnsupportedMediaError, match="application/pdf"):
        load_frames(b"%PDF-1.4", "application/pdf")


def test_corrupt_bytes_raise_a_decode_error() -> None:
    with pytest.raises(MediaDecodeError):
        load_frames(b"not an image at all", "image/png")


def test_truncated_video_raises_a_decode_error() -> None:
    with pytest.raises(MediaDecodeError):
        load_frames(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32, "video/mp4")


class TestLetterbox:
    def test_a_square_image_needs_no_padding(self) -> None:
        canvas = letterbox(gradient(100, 100))

        assert canvas.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)

    def test_the_longest_edge_is_scaled_to_the_canvas(self) -> None:
        canvas = letterbox(gradient(512, 128))

        # 512 -> 256, so 128 -> 64 and 96 rows of padding sit above and below.
        assert canvas.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert np.array_equal(canvas[95], canvas[96])  # padding, replicating row 0
        assert not np.array_equal(canvas[96], canvas[97])  # the image itself

    def test_the_short_edge_is_truncated_not_rounded(self) -> None:
        """
        Ensure the shorter edge is truncated rather than rounded.

        The training transform derives its target size using integer truncation,
        and standard rounding is often one pixel off from this behavior.
        """
        canvas = letterbox(gradient(100, 33))

        # 33 * (256/100) = 84.48 -> 84 rows of image, 86 above and 86 below.
        top = (IMAGE_SIZE - 84) // 2
        assert np.array_equal(canvas[top - 1], canvas[top])  # padding replicates row 0
        assert not np.array_equal(canvas[top], canvas[top + 1])

    def test_padding_replicates_the_edge_rather_than_filling_a_colour(self) -> None:
        canvas = letterbox(gradient(256, 64))

        top = (IMAGE_SIZE - 64) // 2
        # Every padded row above the image equals its first row, exactly.
        assert np.array_equal(canvas[:top], np.repeat(canvas[top : top + 1], top, 0))
        assert not np.array_equal(canvas[0], np.zeros_like(canvas[0]))

    def test_the_image_is_centred(self) -> None:
        """
        Ensure the image is accurately centered on the canvas.

        Padding splits evenly, favoring the top edge by one row when there
        is an odd remainder.
        """
        canvas = letterbox(gradient(256, 101))

        top, bottom = 77, 78  # (256 - 101) // 2, and the rest
        assert np.array_equal(canvas[top - 1], canvas[top])
        assert np.array_equal(canvas[-bottom], canvas[-bottom - 1])
        assert not np.array_equal(canvas[top], canvas[top + 1])

    def test_a_one_pixel_image_survives(self) -> None:
        canvas = letterbox(Image.new("RGB", (1, 1), (7, 8, 9)))

        assert canvas.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert tuple(canvas[0][0]) == (7, 8, 9)


class TestToTensor:
    def test_shape_and_dtype_match_the_graph_input(self) -> None:
        tensor = to_tensor(gradient())

        assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert tensor.dtype == np.float32  # float64 would be rejected by the runtime

    def test_normalisation_uses_the_imagenet_statistics(self) -> None:
        tensor = to_tensor(Image.new("RGB", (16, 16), (0, 0, 0)))

        # A black pixel lands at -mean/std on every channel.
        expected = -np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225])
        assert np.allclose(tensor[0, :, 0, 0], expected, atol=1e-5)

    def test_channels_stay_in_rgb_order(self) -> None:
        """
        Ensure the color channels remain in RGB order.

        A BGR mixup would inadvertently put the red channel last.
        """
        tensor = to_tensor(Image.new("RGB", (16, 16), (255, 0, 0)))

        red, green, blue = tensor[0, :, 0, 0]
        assert red == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-5)
        assert green == pytest.approx(-0.456 / 0.224, abs=1e-5)
        assert blue == pytest.approx(-0.406 / 0.225, abs=1e-5)
        assert red > green and red > blue


class TestVideoFrames:
    def test_frames_are_sampled_up_to_the_requested_count(self, mp4: Any) -> None:
        frames = load_frames(mp4, "video/mp4", max_frames=8)

        assert len(frames) == 8
        assert all(frame.mode == "RGB" for frame in frames)

    def test_sampling_spreads_across_the_clip(self, mp4: Any) -> None:
        """
        Ensure sampled frames are spread evenly across the clip.

        Because the test clip darkens over time, distinct red values
        indicate distinct timestamps.
        """
        frames = load_frames(mp4, "video/mp4", max_frames=4)

        reds = [frame.getpixel((32, 32))[0] for frame in frames]
        assert len(set(reds)) > 1
        assert reds == sorted(reds, reverse=True)

    def test_webm_is_sampled_too(self, webm: Any) -> None:
        frames = load_frames(webm, "video/webm", max_frames=4)

        assert 1 <= len(frames) <= 4

    def test_quicktime_is_sampled_too(self) -> None:
        clip = video_bytes("mov", "libx264", frames=20)

        frames = load_frames(clip, "video/quicktime", max_frames=4)

        assert 1 <= len(frames) <= 4

    def test_a_clip_shorter_than_the_sample_count_yields_fewer(self) -> None:
        clip = video_bytes("mp4", "libx264", frames=2)

        frames = load_frames(clip, "video/mp4", max_frames=8)

        assert 1 <= len(frames) <= 2

    def test_a_single_frame_request_is_honoured(self, mp4: Any) -> None:
        assert len(load_frames(mp4, "video/mp4", max_frames=1)) == 1


class TestSessionResolution:
    def test_the_bundled_model_is_used_by_default(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("GUARD_LOCAL_MODEL_PATH", raising=False)

        path = ModelSession().resolve_path()

        assert path.endswith("model_fp16.onnx")
        assert "guard_local" in path

    def test_the_environment_variable_is_honoured(self, monkeypatch: Any) -> None:
        """Ensure standalone users get the same configuration knob the client uses."""
        bundled = ModelSession()._bundled_model_path()
        monkeypatch.setenv("GUARD_LOCAL_MODEL_PATH", bundled)

        assert ModelSession().resolve_path() == bundled

    def test_the_explicit_argument_wins_over_the_environment(
        self, monkeypatch: Any, tmp_path: Any
    ) -> None:
        bundled = ModelSession()._bundled_model_path()
        monkeypatch.setenv("GUARD_LOCAL_MODEL_PATH", str(tmp_path / "ignored.onnx"))

        assert ModelSession(bundled).resolve_path() == bundled

    def test_a_directory_resolves_to_the_model_inside_it(self) -> None:
        import os

        bundled = ModelSession()._bundled_model_path()
        model_dir = os.path.dirname(os.path.dirname(bundled))

        assert ModelSession(model_dir).resolve_path() == bundled

    def test_a_missing_environment_path_names_where_it_came_from(
        self, monkeypatch: Any, tmp_path: Any
    ) -> None:
        monkeypatch.setenv("GUARD_LOCAL_MODEL_PATH", str(tmp_path / "absent.onnx"))

        with pytest.raises(ModelLoadError, match=r"\$GUARD_LOCAL_MODEL_PATH"):
            ModelSession().resolve_path()

    def test_the_session_is_built_once_and_reused(self) -> None:
        session = ModelSession()

        first, name = session.ensure()

        assert session.ensure() == (first, name)
        assert session._session is first


class TestModelSource:
    def test_reports_every_category(self) -> None:
        """The model is the only source that always has an opinion on all three."""
        found = ModelSource().analyze(png_bytes(), "image/png")

        assert set(found) == set(CATEGORIES)
        assert all(isinstance(result, CategoryResult) for result in found.values())

    def test_every_result_carries_exactly_one_match_naming_the_model(self) -> None:
        found = ModelSource().analyze(png_bytes(), "image/png")

        for category, result in found.items():
            assert [match.source for match in result.matches] == ["model"]
            assert result.matches[0].category == category

    def test_the_result_keeps_more_precision_than_its_match(self) -> None:
        """
        The match rounds for display; the result must not.

        This is the deliberate refinement over the browser extension, which rounds the
        model to a whole percent before weighing it against the other sources.
        """
        found = ModelSource().analyze(jpeg_bytes(64, 48), "image/jpeg")
        result = found["aiGenerated"]

        assert result.confidence == pytest.approx(25.7501, abs=1e-3)
        assert result.matches[0].confidence == 26

    def test_a_confident_score_and_a_quiet_one_point_opposite_ways(self) -> None:
        found = ModelSource().analyze(jpeg_bytes(64, 48), "image/jpeg")

        assert found["aiGenerated"].matches[0].kind == "authentic"
        assert found["violent"].matches[0].kind == "safe"

    def test_max_aggregation_reports_the_worst_frame(self, mp4: Any) -> None:
        worst = ModelSource(video_aggregate="max").analyze(mp4, "video/mp4")
        averaged = ModelSource(video_aggregate="mean").analyze(mp4, "video/mp4")

        assert worst["aiGenerated"].confidence >= averaged["aiGenerated"].confidence

    def test_stills_are_never_sampled_as_clips(self, monkeypatch: Any) -> None:
        seen: list[int] = []

        def spy(data: bytes, media_type: str, *, max_frames: int) -> Any:
            seen.append(max_frames)
            return [Image.new("RGB", (8, 8))]

        monkeypatch.setattr("guard_local.sources.model.source.load_frames", spy)
        ModelSource(video_frames=16).analyze(png_bytes(), "image/png")

        assert seen == [1]


class TestSigmoid:
    @pytest.mark.parametrize(
        ("logit", "expected"),
        [(0.0, 0.5), (-4.310639, 0.0132471), (2.0, 0.8807971)],
    )
    def test_matches_the_logistic_function(self, logit: float, expected: float) -> None:
        assert _sigmoid(logit) == pytest.approx(expected, abs=1e-5)

    def test_does_not_overflow_on_extreme_logits(self) -> None:
        assert _sigmoid(-1000.0) == pytest.approx(0.0)
        assert _sigmoid(1000.0) == pytest.approx(1.0)
