"""
The engine's internal behavior beyond the contract asserted by the client.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from guard_local import LocalDetectorEngine
from guard_local.engine import _sigmoid
from guard_local.exceptions import (
    GuardLocalError,
    MediaDecodeError,
    ModelLoadError,
    UnsupportedMediaError,
)
from guard_local.models import SUPPORTED_MEDIA_TYPES, TASKS

from .conftest import jpeg_bytes, png_bytes


class TestConstruction:
    def test_constructing_loads_nothing(self) -> None:
        """
        Ensure no models are loaded during construction.

        The client reports any constructor failure as a missing installation,
        so model loading problems must not surface here.
        """
        engine = LocalDetectorEngine()

        assert engine._session is None

    def test_a_bad_path_is_not_reported_until_analyze(self, tmp_path: Any) -> None:
        engine = LocalDetectorEngine(tmp_path / "nope.onnx")

        with pytest.raises(ModelLoadError, match="No model file"):
            engine.analyze(png_bytes(), "image/png")

    def test_rejects_a_nonsensical_frame_count(self) -> None:
        with pytest.raises(ValueError, match="video_frames"):
            LocalDetectorEngine(video_frames=0)

    def test_rejects_an_unknown_aggregation(self) -> None:
        with pytest.raises(ValueError, match="video_aggregate"):
            LocalDetectorEngine(video_aggregate="median")


class TestModelResolution:
    def test_the_bundled_model_is_used_by_default(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("GUARD_LOCAL_MODEL_PATH", raising=False)

        path = LocalDetectorEngine()._resolve_model_path()

        assert path.endswith("model_fp16.onnx")
        assert "guard_local" in path

    def test_the_environment_variable_is_honoured(self, monkeypatch: Any) -> None:
        """Ensure standalone users get the same configuration knob the client uses."""
        bundled = LocalDetectorEngine()._bundled_model_path()
        monkeypatch.setenv("GUARD_LOCAL_MODEL_PATH", bundled)

        assert LocalDetectorEngine()._resolve_model_path() == bundled

    def test_the_explicit_argument_wins_over_the_environment(
        self, monkeypatch: Any, tmp_path: Any
    ) -> None:
        bundled = LocalDetectorEngine()._bundled_model_path()
        monkeypatch.setenv("GUARD_LOCAL_MODEL_PATH", str(tmp_path / "ignored.onnx"))

        assert LocalDetectorEngine(bundled)._resolve_model_path() == bundled

    def test_a_directory_resolves_to_the_model_inside_it(self) -> None:
        import os

        bundled = LocalDetectorEngine()._bundled_model_path()
        model_dir = os.path.dirname(os.path.dirname(bundled))

        assert LocalDetectorEngine(model_dir)._resolve_model_path() == bundled

    def test_a_missing_environment_path_names_where_it_came_from(
        self, monkeypatch: Any, tmp_path: Any
    ) -> None:
        monkeypatch.setenv("GUARD_LOCAL_MODEL_PATH", str(tmp_path / "absent.onnx"))

        with pytest.raises(ModelLoadError, match=r"\$GUARD_LOCAL_MODEL_PATH"):
            LocalDetectorEngine()._resolve_model_path()


class TestAnalyze:
    def test_reports_every_task_in_a_stable_order(self, engine: Any) -> None:
        results = engine.analyze(png_bytes(), "image/png")

        assert [entry["label"] for entry in results] == [task.label for task in TASKS]
        assert [entry["label"] for entry in results] == [
            "AI-Generated",
            "Violence",
            "Explicit",
        ]

    def test_carries_the_cloud_descriptions(self, engine: Any) -> None:
        results = engine.analyze(png_bytes(), "image/png")

        assert results[0]["description"] == "Detect AI-generated or manipulated media"

    def test_scores_are_floats_in_the_unit_interval(self, engine: Any) -> None:
        results = engine.analyze(jpeg_bytes(), "image/jpeg")

        for entry in results:
            assert isinstance(entry["score"], float)
            assert 0.0 <= entry["score"] <= 1.0

    def test_emits_no_task_id_so_the_client_derives_one(self, engine: Any) -> None:
        """
        Ensure no task ID is emitted so the client derives it itself.

        Using a UUID5 hash of the label keeps IDs reproducible. Emitting
        an ID directly from the engine would only introduce noise.
        """
        results = engine.analyze(png_bytes(), "image/png")

        assert all("task_id" not in entry for entry in results)

    def test_the_same_bytes_score_the_same(self, engine: Any) -> None:
        data = jpeg_bytes()

        assert engine.analyze(data, "image/jpeg") == engine.analyze(data, "image/jpeg")

    def test_the_session_is_built_once_and_reused(self) -> None:
        engine = LocalDetectorEngine()

        engine.analyze(png_bytes(), "image/png")
        session = engine._session
        engine.analyze(png_bytes(), "image/png")

        assert engine._session is session

    def test_an_unsupported_type_is_rejected_before_decoding(self, engine: Any) -> None:
        with pytest.raises(UnsupportedMediaError, match="application/pdf"):
            engine.analyze(png_bytes(), "application/pdf")

    def test_corrupt_media_raises_a_decode_error_not_a_pil_error(self, engine: Any) -> None:
        with pytest.raises(MediaDecodeError):
            engine.analyze(b"definitely not a png", "image/png")

    @pytest.mark.parametrize("media_type", sorted(SUPPORTED_MEDIA_TYPES))
    def test_no_supported_type_fails_opaquely(self, engine: Any, media_type: str) -> None:
        """
        Ensure supported types do not fail opaquely when mismatched.

        The client forwards all eight supported media types without wrapping
        them. Because of this, offering a PNG as a video must still raise a
        catchable exception rather than failing silently or opaquely.
        """
        with contextlib.suppress(GuardLocalError):
            engine.analyze(png_bytes(), media_type)


class TestVideo:
    def test_a_clip_is_scored_across_frames(self, engine: Any, mp4: Any) -> None:
        results = engine.analyze(mp4, "video/mp4")

        assert len(results) == len(TASKS)
        assert all(0.0 <= entry["score"] <= 1.0 for entry in results)

    def test_max_aggregation_reports_the_worst_frame(self, mp4: Any, monkeypatch: Any) -> None:
        engine = LocalDetectorEngine(video_aggregate="max")

        assert engine._aggregate([0.1, 0.9, 0.3]) == 0.9

    def test_mean_aggregation_averages_the_frames(self) -> None:
        engine = LocalDetectorEngine(video_aggregate="mean")

        assert engine._aggregate([0.2, 0.4, 0.6]) == pytest.approx(0.4)

    def test_the_frame_budget_reaches_the_sampler(self, mp4: Any, monkeypatch: Any) -> None:
        seen = {}
        import guard_local.engine as engine_module

        real = engine_module.load_frames

        def spy(data: bytes, media_type: str, *, max_frames: int) -> Any:
            seen["max_frames"] = max_frames
            return real(data, media_type, max_frames=max_frames)

        monkeypatch.setattr(engine_module, "load_frames", spy)
        LocalDetectorEngine(video_frames=3).analyze(mp4, "video/mp4")

        assert seen["max_frames"] == 3

    def test_stills_are_never_sampled_as_clips(self, engine: Any, monkeypatch: Any) -> None:
        seen = {}
        import guard_local.engine as engine_module

        real = engine_module.load_frames

        def spy(data: bytes, media_type: str, *, max_frames: int) -> Any:
            seen["max_frames"] = max_frames
            return real(data, media_type, max_frames=max_frames)

        monkeypatch.setattr(engine_module, "load_frames", spy)
        engine.analyze(png_bytes(), "image/png")

        assert seen["max_frames"] == 1


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


def test_golden_scores_for_a_deterministic_image(engine: Any) -> None:
    """
    Pin the entire pipeline to ensure deterministic scoring.

    A wrong resample filter, a dropped EXIF transpose, or a transposed channel order
    will all change these numbers without changing the output shape. Pinning these
    golden scores is the only way to catch subtle pipeline regressions.
    """
    results = {
        entry["label"]: entry["score"]
        for entry in engine.analyze(jpeg_bytes(64, 48), "image/jpeg")
    }

    assert results["AI-Generated"] == pytest.approx(GOLDEN["AI-Generated"], abs=1e-4)
    assert results["Violence"] == pytest.approx(GOLDEN["Violence"], abs=1e-4)
    assert results["Explicit"] == pytest.approx(GOLDEN["Explicit"], abs=1e-4)


#: Recorded from the bundled model; see the docstring above for why they are pinned.
GOLDEN = {
    "AI-Generated": 0.257501,
    "Violence": 0.000646,
    "Explicit": 0.033055,
}
