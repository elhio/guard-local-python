"""
Walks a manifest store to extract the chain of manifests it holds.

The store is a flat map of manifests paired with a pointer to the active one. What
matters most is the chain. This includes the active manifest and every ingredient it was
derived from recursively. This traversal is essential because the strongest evidence of
AI origin is usually recorded at the moment of creation in the first ingredient.
Meanwhile, the active manifest at the top often records nothing more interesting than a
simple crop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .manifest_actions import Manifest

__all__ = ["get_manifest_chain"]


def get_manifest_chain(store: Optional[Dict[str, Any]]) -> List[Manifest]:
    """
    Flatten a manifest store into the active manifest and its ancestry.

    Args:
        store: The parsed manifest store, or `None` when the asset carries none.

    Returns:
        A list containing the active manifest first, followed by every manifest it
        derives from in depth-first order. Each manifest appears only once. A visited
        set guards against an ingredient graph that might loop back on itself.
    """
    manifests = (store or {}).get("manifests") or {}
    if not manifests:
        return []

    chain: List[Manifest] = []
    visited: Set[str] = set()

    def visit(label: Optional[str]) -> None:
        if not label or label in visited:
            return
        manifest = manifests.get(label)
        if not manifest:
            return

        visited.add(label)
        chain.append(manifest)

        for ingredient in manifest.get("ingredients") or []:
            visit(ingredient.get("active_manifest"))

    visit(store.get("active_manifest") if store else None)
    return chain
