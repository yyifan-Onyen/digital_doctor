"""Load the declarative state, action, and phase definitions for this skill."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


PACKAGE_DIR = Path(__file__).resolve().parent


def _load_json(name: str) -> Dict[str, object]:
    with (PACKAGE_DIR / name).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


STATE_SCHEMA = _load_json("state_schema.json")
ACTION_CONFIG = _load_json("actions.json")
PHASE_CONFIG = _load_json("phase_graph.json")

FORMULATION_FIELDS: List[Tuple[str, str]] = [
    (str(name), str(spec.get("title", name)))
    for name, spec in dict(STATE_SCHEMA.get("properties", {})).items()
    if isinstance(spec, dict)
]
REQUIRED_TREATMENT_FIELDS: Tuple[str, ...] = tuple(
    str(item) for item in STATE_SCHEMA.get("x-required-for-treatment", [])
)
REQUIRED_PREREQUISITE_SLUGS: Tuple[str, ...] = tuple(
    str(item) for item in PHASE_CONFIG.get("required_prerequisite_slugs", [])
)
PHASE_SPECS: List[Dict[str, object]] = [
    dict(item) for item in PHASE_CONFIG.get("phases", []) if isinstance(item, dict)
]
ACTION_GUIDANCE: Dict[str, str] = {
    str(name): str(guidance)
    for name, guidance in dict(ACTION_CONFIG.get("actions", {})).items()
}
VALID_RESPONSE_MOVES = frozenset(ACTION_GUIDANCE)
VALID_RESPONSE_DEPTHS = frozenset(str(item) for item in ACTION_CONFIG.get("depths", []))


__all__ = [
    "ACTION_CONFIG",
    "ACTION_GUIDANCE",
    "FORMULATION_FIELDS",
    "PACKAGE_DIR",
    "PHASE_CONFIG",
    "PHASE_SPECS",
    "REQUIRED_PREREQUISITE_SLUGS",
    "REQUIRED_TREATMENT_FIELDS",
    "STATE_SCHEMA",
    "VALID_RESPONSE_DEPTHS",
    "VALID_RESPONSE_MOVES",
]
