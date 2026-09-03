"""Stable, JSON-friendly contracts shared by the harness and clinical skills."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


HARNESS_VERSION = "1.0.0"


@dataclass(frozen=True)
class SkillManifest:
    skill_id: str
    name: str
    version: str
    description: str
    entry_point: str
    capabilities: Tuple[str, ...] = ()
    primary: bool = True
    bundle_checksum: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "entry_point": self.entry_point,
            "capabilities": list(self.capabilities),
            "primary": self.primary,
            "bundle_checksum": self.bundle_checksum or self.metadata_checksum(),
        }

    def metadata_checksum(self) -> str:
        payload = {
            "skill_id": self.skill_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "entry_point": self.entry_point,
            "capabilities": list(self.capabilities),
            "primary": self.primary,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TurnContext:
    session_id: str
    episode_id: str
    user_id: str
    query: str
    history: str
    turn_index: int
    memory_recall: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class StateDelta:
    updates: List[Dict[str, object]] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"updates": list(self.updates), "metadata": dict(self.metadata)}


@dataclass
class ActionPlan:
    mode: str
    action: str
    depth: str
    reason: str
    allowed_actions: List[str] = field(default_factory=list)
    authorized: bool = True
    authorization_reason: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ActionPlan":
        action = str(value.get("response_move", value.get("action", "assess")))
        return cls(
            mode=str(value.get("mode", "analysis")),
            action=action,
            depth=str(value.get("depth", "standard")),
            reason=str(value.get("reason", "")),
            allowed_actions=[str(item) for item in value.get("allowed_actions", [])],
        )

    def to_route_dict(self) -> Dict[str, str]:
        return {
            "mode": self.mode,
            "response_move": self.action,
            "depth": self.depth,
            "reason": self.reason,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            **self.to_route_dict(),
            "action": self.action,
            "allowed_actions": list(self.allowed_actions),
            "authorized": self.authorized,
            "authorization_reason": self.authorization_reason,
        }


@dataclass
class EvidenceBundle:
    transcript_refs: List[str] = field(default_factory=list)
    knowledge_hits: List[Dict[str, object]] = field(default_factory=list)
    helper_query: str = ""
    helper_answer: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "transcript_refs": list(self.transcript_refs),
            "knowledge_hits": list(self.knowledge_hits),
            "helper_query": self.helper_query,
            "helper_answer": self.helper_answer,
        }


@dataclass(frozen=True)
class GenerationSpec:
    stage: str
    prompt: str
    json_mode: bool = False
    candidate_name: str = "combined"
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class SkillVerdict:
    action: str = "allow"
    risk_level: str = "none"
    categories: List[str] = field(default_factory=list)
    rationale: str = ""
    final_reply: str = ""
    changed: bool = False
    escalated: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HarnessIdentity:
    harness_version: str
    skill_id: str
    skill_version: str
    skill_checksum: str
    model_adapter: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def compact_mapping(value: Optional[Mapping[str, object]]) -> Dict[str, object]:
    return dict(value or {})


__all__ = [
    "ActionPlan",
    "EvidenceBundle",
    "GenerationSpec",
    "HARNESS_VERSION",
    "HarnessIdentity",
    "SkillManifest",
    "SkillVerdict",
    "StateDelta",
    "TurnContext",
]
