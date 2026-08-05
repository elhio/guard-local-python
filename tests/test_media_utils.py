"""
Decodes media bytes and transforms them for the model.

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

from guard_local.exceptions import MediaDecodeError, UnsupportedMediaError
from guard_local.media_utils import IMAGE_SIZE, letterbox, load_frames, to_tensor

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
def test_every_still_format_decodes_to_one_rgb_frame(data: bytes, media_type: str) -> None:
    frames = load_frames(data, media_type)

    assert len(frames) == 1
    assert frames[0].mode == "RGB"


def test_animated_gif_yields_its_first_frame() -> None:
    """Ensure animated GIFs yield only their first frame to match the browser extension."""
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
        assert np.array_equal(canvas[:top], np.repeat(canvas[top: top + 1], top, 0))
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
