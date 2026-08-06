"""
The rules that turn a pile of signals into a verdict.

These are the only place the engine decides what evidence adds up to, so the arithmetic
is pinned here rather than inferred from an end-to-end score.
"""

from __future__ import annotations

import pytest

from guard_local.detection import (
    AI_GENERATED,
    CATEGORIES,
    DETECTION_THRESHOLDS,
    EXPLICIT,
    VIOLENT,
    CategoryResult,
    Signal,
    SignalMatch,
    bucket_matches_by_category,
    combine_category_results,
    merge_category_results,
    passes_threshold,
)


def match(category: str, confidence: int, identifier: str = "test") -> SignalMatch:
    return SignalMatch.of(
        Signal(
            id=identifier,
            category=category,
            label="Test",
            description="A test signal",
            confidence=confidence,
        ),
        evidence="because",
        source="metadata",
    )


class TestThresholds:
    def test_ai_generation_is_held_to_a_higher_bar(self) -> None:
        assert DETECTION_THRESHOLDS[AI_GENERATED] > DETECTION_THRESHOLDS[VIOLENT]

    @pytest.mark.parametrize(
        ("category", "score", "expected"),
        [
            (AI_GENERATED, 89, False),
            (AI_GENERATED, 90, True),
            (VIOLENT, 69, False),
            (VIOLENT, 70, True),
            (EXPLICIT, 70, True),
        ],
    )
    def test_the_threshold_is_inclusive(
        self, category: str, score: int, expected: bool
    ) -> None:
        assert passes_threshold(category, score) is expected

    def test_an_unknown_category_never_passes(self) -> None:
        """A typo must fail closed rather than flag everything it touches."""
        assert passes_threshold("nonsense", 100) is False


class TestBucketing:
    def test_categories_with_no_matches_are_absent(self) -> None:
        """
        Ensure untouched categories are omitted rather than reported as clear.

        An absent category means nothing was found, which is different from a category
        that was examined and came back at zero.
        """
        buckets = bucket_matches_by_category([match(VIOLENT, 80)])

        assert set(buckets) == {VIOLENT}

    def test_the_loudest_match_sets_the_confidence(self) -> None:
        buckets = bucket_matches_by_category(
            [match(AI_GENERATED, 20), match(AI_GENERATED, 95), match(AI_GENERATED, 60)]
        )

        assert buckets[AI_GENERATED].confidence == 95
        assert buckets[AI_GENERATED].detected is True

    def test_matches_are_ordered_loudest_first(self) -> None:
        buckets = bucket_matches_by_category(
            [match(AI_GENERATED, 20), match(AI_GENERATED, 95), match(AI_GENERATED, 60)]
        )

        assert [m.confidence for m in buckets[AI_GENERATED].matches] == [95, 60, 20]

    def test_a_weak_match_alone_does_not_detect(self) -> None:
        buckets = bucket_matches_by_category([match(AI_GENERATED, 25)])

        assert buckets[AI_GENERATED].confidence == 25
        assert buckets[AI_GENERATED].detected is False

    def test_each_category_is_judged_by_its_own_threshold(self) -> None:
        buckets = bucket_matches_by_category(
            [match(AI_GENERATED, 80), match(VIOLENT, 80)]
        )

        assert buckets[AI_GENERATED].detected is False  # needs 90
        assert buckets[VIOLENT].detected is True  # needs 70

    def test_no_matches_yield_no_buckets(self) -> None:
        assert bucket_matches_by_category([]) == {}

    def test_buckets_do_not_share_a_match_list(self) -> None:
        """A shared mutable default would leak one category's evidence into another."""
        first = bucket_matches_by_category([match(VIOLENT, 80)])
        second = bucket_matches_by_category([match(EXPLICIT, 80)])

        assert len(first[VIOLENT].matches) == 1
        assert len(second[EXPLICIT].matches) == 1


class TestCombining:
    def test_every_source_contributes_its_matches(self) -> None:
        combined = combine_category_results(
            AI_GENERATED,
            [
                CategoryResult(matches=[match(AI_GENERATED, 30, "a")]),
                CategoryResult(matches=[match(AI_GENERATED, 95, "b")]),
            ],
        )

        assert [m.id for m in combined.matches] == ["b", "a"]

    def test_the_loudest_source_decides(self) -> None:
        """
        Ensure a strong signal is not diluted by weak ones.

        Signals are alternative pieces of evidence rather than samples to average, so
        one signed provenance claim outweighs any number of weak heuristics.
        """
        combined = combine_category_results(
            AI_GENERATED,
            [
                CategoryResult(matches=[match(AI_GENERATED, 95)]),
                CategoryResult(matches=[match(AI_GENERATED, 5)]),
                CategoryResult(matches=[match(AI_GENERATED, 10)]),
            ],
        )

        assert combined.confidence == 95
        assert combined.detected is True

    def test_combining_nothing_is_a_clean_zero(self) -> None:
        combined = combine_category_results(AI_GENERATED, [])

        assert combined.confidence == 0
        assert combined.detected is False
        assert combined.matches == []


class TestSignalMatch:
    def test_a_match_carries_every_field_of_its_signal(self) -> None:
        signal = Signal(
            id="x", category=VIOLENT, label="L", description="D", confidence=70
        )

        found = SignalMatch.of(signal, "the evidence", "metadata")

        assert (found.id, found.category, found.confidence) == ("x", VIOLENT, 70)
        assert found.evidence == "the evidence"
        assert found.source == "metadata"

    def test_a_match_renders_to_a_plain_dictionary(self) -> None:
        """Callers read results without importing anything from this package."""
        rendered = match(VIOLENT, 70).to_dict()

        assert rendered["confidence"] == 70
        assert rendered["source"] == "metadata"
        assert set(rendered) == {
            "id",
            "category",
            "label",
            "description",
            "confidence",
            "kind",
            "evidence",
            "source",
        }


class TestSourceProtocol:
    """
    Every source answers in one shape, which is what lets the engine stay ignorant.
    """

    def test_all_three_sources_satisfy_the_protocol(self) -> None:
        from guard_local.sources import C2paSource, MetadataSource, ModelSource, Source

        for source in (C2paSource(), MetadataSource(), ModelSource()):
            assert isinstance(source, Source)

    def test_every_source_names_itself_as_its_matches_do(self) -> None:
        from guard_local.sources import C2paSource, MetadataSource, ModelSource

        names = [
            source.name for source in (C2paSource(), MetadataSource(), ModelSource())
        ]

        assert names == ["c2pa", "metadata", "model"]

    def test_every_source_answers_with_results_keyed_by_category(self) -> None:
        from guard_local.sources import C2paSource, MetadataSource, ModelSource

        from .conftest import png_bytes

        data = png_bytes()
        for source in (C2paSource(), MetadataSource(), ModelSource()):
            found = source.analyze(data, "image/png")

            assert isinstance(found, dict)
            assert set(found) <= set(CATEGORIES)
            assert all(isinstance(result, CategoryResult) for result in found.values())


class TestMergingSources:
    def test_categories_from_different_sources_are_all_kept(self) -> None:
        merged = merge_category_results(
            [
                {AI_GENERATED: CategoryResult(confidence=90, matches=[])},
                {VIOLENT: CategoryResult(confidence=70, matches=[])},
            ]
        )

        assert set(merged) == {AI_GENERATED, VIOLENT}

    def test_the_loudest_source_wins_a_shared_category(self) -> None:
        merged = merge_category_results(
            [
                {AI_GENERATED: CategoryResult(confidence=25, matches=[])},
                {AI_GENERATED: CategoryResult(confidence=90, matches=[])},
            ]
        )

        assert merged[AI_GENERATED].confidence == 90
        assert merged[AI_GENERATED].detected is True

    def test_a_precise_confidence_survives_the_merge(self) -> None:
        """
        Only the model reports a confidence finer than a whole percent.

        Deriving the merged confidence from the matches would silently round it away,
        which is the one thing this engine does differently from the browser extension.
        """
        merged = merge_category_results(
            [{AI_GENERATED: CategoryResult(confidence=96.29, matches=[])}]
        )

        assert merged[AI_GENERATED].confidence == pytest.approx(96.29)

    def test_nothing_found_is_an_empty_mapping(self) -> None:
        assert merge_category_results([{}, {}]) == {}
