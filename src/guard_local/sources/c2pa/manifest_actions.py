"""
Extracts the recorded actions from a manifest.

Actions represent the history of what was done to an asset and by what tool or agent.
They are stored inside assertions rather than at the top level of a manifest. Because
the assertion carrying them is versioned, the parser must accept formats like both
`c2pa.actions` and `c2pa.actions.v2`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["Manifest", "get_actions", "get_software_agent_name"]

#: A manifest as parsed from the reader's JSON output. This is kept as a plain mapping
#: rather than a strict dataclass so that unfamiliar fields still survive to be
#: searched.
Manifest = Dict[str, Any]


def get_actions(manifest: Manifest) -> List[Dict[str, Any]]:
    """
    Collect every action recorded in a manifest.

    Args:
        manifest: One parsed manifest.

    Returns:
        The actions from every actions assertion flattened into a single list.
    """
    actions: List[Dict[str, Any]] = []
    for assertion in manifest.get("assertions") or []:
        label = assertion.get("label", "")
        if label == "c2pa.actions" or label.startswith("c2pa.actions."):
            data = assertion.get("data") or {}
            actions.extend(data.get("actions") or [])
    return [action for action in actions if isinstance(action, dict)]


def get_software_agent_name(software_agent: Any) -> Optional[str]:
    """
    Resolve the name of the tool or agent that performed an action.

    Args:
        software_agent: The `softwareAgent` field of an action. Older writers record
            this as a bare string, while newer ones use a structured generator
            description.

    Returns:
        The name of the agent, or `None` if the action does not provide one.
    """
    if not software_agent:
        return None
    if isinstance(software_agent, str):
        return software_agent
    if isinstance(software_agent, dict):
        name = software_agent.get("name")
        return str(name) if name else None
    return None
