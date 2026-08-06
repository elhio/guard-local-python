"""
The shared vocabulary used by every detection layer.

A signal represents a single rule, heuristic, or model output. A match combines that
signal with the specific data that triggered it. Keeping all three layers aligned on
this single shape allows the engine to merge a C2PA assertion, an EXIF tag, and a model
logit into a unified verdict without any layer needing to know about the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

__all__ = [
    "CATEGORIES",
    "AI_GENERATED",
    "CategoryResult",
    "EXPLICIT",
    "Signal",
    "SignalMatch",
    "VIOLENT",
]

#: The three moderation categories.
AI_GENERATED = "aiGenerated"
VIOLENT = "violent"
EXPLICIT = "explicit"

CATEGORIES = (AI_GENERATED, VIOLENT, EXPLICIT)


@dataclass(frozen=True)
class Signal:
    """
    One rule that argues for or against a category when triggered.

    Attributes:
        id: A stable identifier that is unique across every layer. Callers
            may branch on this, making it as much of a public API as the
            task labels.
        category: The category this signal evaluates.
        label: A short plain text name.
        description: An explanation of what the signal means and why it
            points in the direction it does.
        confidence: How strongly this signal alone implies the category,
            measured from 0 to 100.
        kind: The direction of the evidence. The values `"authentic"` and
            `"safe"` mark signals that argue against their category, while
            all other values argue for it.
    """

    id: str
    category: str
    label: str
    description: str
    confidence: int
    kind: Optional[str] = None


@dataclass(frozen=True)
class SignalMatch(Signal):
    """
    A triggered signal paired with the data that triggered it.

    Attributes:
        evidence: The specific tag, assertion, or score found in the media that
            triggered this signal.
        source: The layer that produced the match, such as `"metadata"`, `"c2pa"`, or
            `"model"`.
    """

    evidence: str = ""
    source: str = ""

    @classmethod
    def of(cls, signal: Signal, evidence: str, source: str) -> SignalMatch:
        """
        Build a match from a triggered signal.

        Args:
            signal: The catalog entry being reported.
            evidence: The specific data found in the media.
            source: The layer reporting the match.

        Returns:
            A match carrying every field of the original signal along with
            the new evidence.
        """
        return cls(
            id=signal.id,
            category=signal.category,
            label=signal.label,
            description=signal.description,
            confidence=signal.confidence,
            kind=signal.kind,
            evidence=evidence,
            source=source,
        )

    def to_dict(self) -> Dict[str, Union[str, int, None]]:
        """
        Render this match into a dictionary for the engine's return value.

        Returns:
            A plain dictionary so callers never need to import this class just to read
            a result.
        """
        return {
            "id": self.id,
            "category": self.category,
            "label": self.label,
            "description": self.description,
            "confidence": self.confidence,
            "kind": self.kind,
            "evidence": self.evidence,
            "source": self.source,
        }


@dataclass
class CategoryResult:
    """
    The verdict for a single category and all the evidence supporting it.

    Attributes:
        detected: True if the `confidence` clears the threshold for this category.
        confidence: The highest confidence among the matches, or 0.0 if none
            fired. This is a float rather than an integer so a source can
            express its confidence more precisely than a whole percent. Only
            the model does this. Its matches round to a percent for display
            while the result it reports keeps the exact probability the model
            produced.
        matches: Every match found ordered by confidence, highest first.
    """

    detected: bool = False
    confidence: float = 0.0
    matches: List[SignalMatch] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Fall back to the loudest match when no confidence was provided.

        A result that carries matches but claims no confidence of its own would
        otherwise read as zero, which is never what the caller intended. Only a source
        that knows better than its own matches needs to pass one in, and currently, only
        the model does this.
        """
        if not self.confidence and self.matches:
            self.confidence = max(match.confidence for match in self.matches)
