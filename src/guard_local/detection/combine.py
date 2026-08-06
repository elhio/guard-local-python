"""
Folds many independent matches into one verdict per category.

Every detection layer produces a flat list of matches. The functions in this module are
the only places that decide what a collection of matches adds up to, and they all
resolve it the same way. The highest confidence match always wins. This is deliberate
because detection signals here are alternative pieces of evidence rather than
independent observations that can be averaged. For example, a signed `c2pa.ai_generated`
action does not become less true just because ten weak heuristics disagree with it.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from .thresholds import passes_threshold
from .types import CategoryResult, SignalMatch

__all__ = [
    "bucket_matches_by_category",
    "combine_category_results",
    "merge_category_results",
]


def bucket_matches_by_category(
    matches: Iterable[SignalMatch],
) -> Dict[str, CategoryResult]:
    """
    Group a flat list of matches into one result per category.

    Args:
        matches: Matches from any layer, in any order, spanning any categories.

    Returns:
        A result per category that actually matched. Categories with no matches are
        omitted rather than returned empty. This allows a caller to easily distinguish
        between a category where nothing was found and a category that was never
        evaluated.
    """
    buckets: Dict[str, CategoryResult] = {}
    for match in matches:
        buckets.setdefault(match.category, CategoryResult()).matches.append(match)

    for category, bucket in buckets.items():
        bucket.matches.sort(key=lambda match: match.confidence, reverse=True)
        bucket.confidence = bucket.matches[0].confidence if bucket.matches else 0
        bucket.detected = passes_threshold(category, bucket.confidence)

    return buckets


def combine_category_results(
    category: str, results: Iterable[CategoryResult]
) -> CategoryResult:
    """
    Merge several results for the same category into a single result.

    Args:
        category: The category all the results describe.
        results: The per-layer results to merge.

    Returns:
        One result holding every match ordered by confidence, with the final verdict
        taken from the match with the highest confidence.
    """
    collected = list(results)

    matches: List[SignalMatch] = [
        match for result in collected for match in result.matches
    ]
    matches.sort(key=lambda match: match.confidence, reverse=True)

    confidence = max((result.confidence for result in collected), default=0.0)
    return CategoryResult(
        detected=passes_threshold(category, confidence),
        confidence=confidence,
        matches=matches,
    )


def merge_category_results(
    results: Iterable[Dict[str, CategoryResult]],
) -> Dict[str, CategoryResult]:
    """
    Fold what several sources found into one result per category.

    Args:
        results: One mapping per source, exactly as each source's `analyze`
            method returns it. Because sources only provide results for the
            categories they know about, the mappings will routinely disagree
            on which keys they carry.

    Returns:
        A result per category that any source spoke to, merged across every
        source that provided a match. Categories that no source examined
        remain absent from the final dictionary.
    """
    collected = list(results)

    buckets: Dict[str, List[CategoryResult]] = {}
    for result in collected:
        for category, found in result.items():
            buckets.setdefault(category, []).append(found)

    return {
        category: combine_category_results(category, found)
        for category, found in buckets.items()
    }
