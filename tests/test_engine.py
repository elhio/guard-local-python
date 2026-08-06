"""
The engine's internal behavior beyond the contract asserted by the client.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from guard_local import LocalDetectorEngine
from guard_local.exceptions import (
    GuardLocalError,
    MediaDecodeError,
    ModelLoadError,
    UnsupportedMediaError,
)
from guard_local.tasks import SUPPORTED_MEDIA_TYPES, TASKS

from .conftest import jpeg_bytes, png_bytes, png_with_text


class TestConstruction:
    def test_constructing_loads_nothing(self) -> None:
        """
        Ensure no models are loaded during construction.

        The client reports any constructor failure as a missing installation,
        so model loading problems must not surface here.
        """
        engine = LocalDetectorEngine()

        assert engine.model is not None
        assert engine.model.session._session is None

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

    def test_rejects_having_every_layer_turned_off(self) -> None:
        """An engine with nothing enabled would report zero for everything, silently."""
        with pytest.raises(ValueError, match="use_model"):
            LocalDetectorEngine(use_model=False, use_metadata=False, use_c2pa=False)


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

    def test_an_unsupported_type_is_rejected_before_decoding(self, engine: Any) -> None:
        with pytest.raises(UnsupportedMediaError, match="application/pdf"):
            engine.analyze(png_bytes(), "application/pdf")

    def test_corrupt_media_raises_a_decode_error_not_a_pil_error(
        self, engine: Any
    ) -> None:
        with pytest.raises(MediaDecodeError):
            engine.analyze(b"definitely not a png", "image/png")

    @pytest.mark.parametrize("media_type", sorted(SUPPORTED_MEDIA_TYPES))
    def test_no_supported_type_fails_opaquely(
        self, engine: Any, media_type: str
    ) -> None:
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

    def test_the_video_knobs_reach_the_model_source(self, mp4: Any) -> None:
        """The engine owns the knobs; the model source is what acts on them."""
        engine = LocalDetectorEngine(video_frames=3, video_aggregate="mean")

        assert engine.model is not None
        assert engine.model.video_frames == 3
        assert engine.model.video_aggregate == "mean"

    def test_the_frame_budget_reaches_the_sampler(
        self, mp4: Any, monkeypatch: Any
    ) -> None:
        seen = {}
        import guard_local.sources.model.source as source_module

        real = source_module.load_frames

        def spy(data: bytes, media_type: str, *, max_frames: int) -> Any:
            seen["max_frames"] = max_frames
            return real(data, media_type, max_frames=max_frames)

        monkeypatch.setattr(source_module, "load_frames", spy)
        LocalDetectorEngine(video_frames=3).analyze(mp4, "video/mp4")

        assert seen["max_frames"] == 3


class TestEvidenceMerging:
    def test_metadata_outvotes_a_quiet_model(self, engine: Any) -> None:
        """
        Ensure strong metadata decides the verdict when the model sees nothing.

        This is the whole point of the merge. A flat image carries no visual tell, so
        the model rates it near zero, while its own text chunk states the prompt and
        sampler it was generated with.
        """
        data = png_with_text(parameters="Steps: 20, Sampler: Euler a, Seed: 12")

        ai = engine.analyze(data, "image/png")[0]

        assert ai["score"] == pytest.approx(0.90)
        assert ai["detected"] is True
        assert ai["matches"][0]["id"] == "pngtext.source.terms"

    def test_a_confident_model_outvotes_weak_metadata(self, engine: Any) -> None:
        results = LocalDetectorEngine(use_metadata=False, use_c2pa=False).analyze(
            png_bytes(), "image/png"
        )
        model_score = results[0]["score"]

        merged = engine.analyze(png_bytes(), "image/png")[0]

        assert merged["score"] == pytest.approx(model_score)

    def test_the_model_keeps_its_precision_when_it_wins(self, engine: Any) -> None:
        """
        Ensure merging does not round the model's score to a whole percent.

        The browser extension compares on integer percents, which is fine for a badge
        but would throw away precision a caller can never get back. Comparing on the
        same scale picks the same winner without that loss.
        """
        score = engine.analyze(jpeg_bytes(), "image/jpeg")[0]["score"]

        assert score != pytest.approx(round(score * 100) / 100, abs=1e-9)

    def test_the_weak_jfif_floor_does_raise_a_quiet_score(self) -> None:
        """
        Ensure the browser extension's weak floors are matched exactly.

        `jfif.present` is confidence 10, so under a highest-wins merge every JPEG
        floors at 0.10 for AI generation even when the model is certain it is a real
        photo. That is the extension's behaviour, and it is pinned here so the choice
        stays a decision rather than a surprise.
        """
        quiet = LocalDetectorEngine(use_model=False)

        ai = quiet.analyze(jpeg_bytes(), "image/jpeg")[0]

        assert ai["score"] == pytest.approx(0.10)
        assert ai["matches"][0]["id"] == "jfif.present"

    def test_every_layer_names_itself_in_the_evidence(self, engine: Any) -> None:
        data = png_with_text(parameters="Steps: 20, Seed: 12")

        ai = engine.analyze(data, "image/png")[0]

        assert {match["source"] for match in ai["matches"]} == {"metadata", "model"}

    def test_evidence_is_ordered_loudest_first(self, engine: Any) -> None:
        data = png_with_text(parameters="Steps: 20, Sampler: Euler a")

        ai = engine.analyze(data, "image/png")[0]
        confidences = [match["confidence"] for match in ai["matches"]]

        assert confidences == sorted(confidences, reverse=True)

    def test_detected_follows_the_category_threshold(self, engine: Any) -> None:
        """AI generation needs 90; the same confidence would flag the other two."""
        quiet = LocalDetectorEngine(use_model=False)

        results = quiet.analyze(png_bytes(1024, 1024), "image/png")

        assert results[0]["score"] == pytest.approx(0.25)
        assert results[0]["detected"] is False

    def test_a_task_with_nothing_to_report_is_a_clean_zero(self) -> None:
        quiet = LocalDetectorEngine(use_model=False)

        violence = quiet.analyze(png_bytes(), "image/png")[1]

        assert violence["score"] == 0.0
        assert violence["detected"] is False
        assert violence["matches"] == []


class TestLayerToggles:
    def test_turning_the_model_off_never_builds_a_session(self) -> None:
        """Ensure a metadata-only scan costs no ONNX session."""
        quiet = LocalDetectorEngine(use_model=False)

        quiet.analyze(jpeg_bytes(), "image/jpeg")

        assert quiet.model is None
        assert [source.name for source in quiet.sources] == ["c2pa", "metadata"]

    def test_turning_metadata_off_drops_its_signals(self) -> None:
        data = png_with_text(parameters="Steps: 20, Sampler: Euler a")

        without = LocalDetectorEngine(use_metadata=False).analyze(data, "image/png")[0]

        assert all(match["source"] != "metadata" for match in without["matches"])

    def test_turning_c2pa_off_still_reports_metadata(self) -> None:
        data = png_with_text(parameters="Steps: 20, Sampler: Euler a")

        ai = LocalDetectorEngine(use_c2pa=False, use_model=False).analyze(
            data, "image/png"
        )[0]

        assert ai["score"] == pytest.approx(0.90)

    def test_corrupt_media_still_raises_when_the_model_is_on(self) -> None:
        """Ensure a decode failure is not swallowed by the evidence layers."""
        with pytest.raises(MediaDecodeError):
            LocalDetectorEngine().analyze(b"not a png", "image/png")


class TestVideoEvidence:
    def test_metadata_is_not_read_from_a_container(self, mp4: Any) -> None:
        """A clip's provenance lives in its manifest, not in EXIF."""
        results = LocalDetectorEngine(use_model=False).analyze(mp4, "video/mp4")

        assert all(entry["matches"] == [] for entry in results)

    def test_a_clip_is_still_scored_by_the_model(self, engine: Any, mp4: Any) -> None:
        results = engine.analyze(mp4, "video/mp4")

        assert all(entry["matches"] for entry in results)
        assert all(entry["matches"][0]["source"] == "model" for entry in results)


def test_golden_scores_for_a_deterministic_image(engine: Any) -> None:
    """Pin the entire pipeline to ensure deterministic scoring."""
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
