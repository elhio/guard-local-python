"""
The vocabulary shared by every detection layer.

The metadata reader, the C2PA reader, and the vision model all produce `SignalMatch`
values against the same categories. The combine logic then determines what a collection
of these matches means in aggregate. Nothing in this module reads media directly or
imports heavy dependencies.
"""

from __future__ import annotations

from .combine import (
    bucket_matches_by_category,
    combine_category_results,
    merge_category_results,
)
from .thresholds import DETECTION_THRESHOLDS, passes_threshold
from .types import (
    AI_GENERATED,
    CATEGORIES,
    EXPLICIT,
    VIOLENT,
    CategoryResult,
    Signal,
    SignalMatch,
)

__all__ = [
    "AI_GENERATED",
    "CATEGORIES",
    "DETECTION_THRESHOLDS",
    "EXPLICIT",
    "VIOLENT",
    "CategoryResult",
    "Signal",
    "SignalMatch",
    "bucket_matches_by_category",
    "combine_category_results",
    "merge_category_results",
    "passes_threshold",
]
