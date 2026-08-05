"""
Fixture builders for the test suite.

Every sample is generated at import time rather than committed. This ensures the suite
carries no binary fixtures and each test can define exactly what it feeds into the
engine.
"""

from __future__ import annotations

import io

import av
import pytest
from PIL import Image


def gradient(width: int = 64, height: int = 48) -> Image.Image:
    """
    Generate a deterministic, non-uniform image.

    A flat color would make a broken resize or a dropped EXIF transpose
    invisible because every pixel would be identical.
    """
    image = Image.new("RGB", (width, height))
    image.putdata(
        [
            ((x * 4) % 256, (y * 4) % 256, (x * y) % 256)
            for y in range(height)
            for x in range(width)
        ]
    )
    return image


def encode(image: Image.Image, fmt: str, **kwargs) -> bytes:
    """Encode a Pillow image into bytes using the specified format."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def png_bytes(width: int = 64, height: int = 48) -> bytes:
    """Generate a standard PNG test image."""
    return encode(gradient(width, height), "PNG")


def jpeg_bytes(width: int = 64, height: int = 48) -> bytes:
    """Generate a standard JPEG test image."""
    return encode(gradient(width, height), "JPEG")


def webp_bytes() -> bytes:
    """Generate a standard WEBP test image."""
    return encode(gradient(), "WEBP")


def gif_bytes(frames: int = 3) -> bytes:
    """Generate an animated GIF whose first frame is explicitly red."""
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    images = [Image.new("RGB", (32, 32), colours[i % 3]) for i in range(frames)]
    buffer = io.BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


def heic_bytes() -> bytes:
    """Generate a standard HEIC test image."""
    return encode(gradient(), "HEIF")


def exif_rotated_jpeg() -> bytes:
    """Create a landscape JPEG tagged with orientation 6 (portrait)."""
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW
    return encode(gradient(64, 32), "JPEG", exif=exif)


def video_bytes(
    container_format: str = "mp4",
    codec: str = "libx264",
    frames: int = 30,
    size: tuple = (64, 64),
) -> bytes:
    """Create a clip whose frames darken chronologically from red to black."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=container_format) as container:
        stream = container.add_stream(codec, rate=10)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            image = Image.new("RGB", size, (255 - index * 8, 0, 0))
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def mp4() -> bytes:
    """Provide a standard MP4 video fixture encoded with H.264."""
    return video_bytes("mp4", "libx264")


@pytest.fixture(scope="session")
def webm() -> bytes:
    """Provide a standard WebM video fixture encoded with VP8."""
    return video_bytes("webm", "libvpx")


@pytest.fixture(scope="session")
def engine():
    """Provide one detection engine instance for the entire test session."""
    from guard_local import LocalDetectorEngine

    return LocalDetectorEngine()
