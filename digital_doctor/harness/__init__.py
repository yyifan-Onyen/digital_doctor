"""Execution harness for versioned clinical skills."""

from .adapters import CallableModelAdapter, HttpModelAdapter, ModelAdapter, OPSDModelAdapter, SFTModelAdapter
from .contracts import (
    ActionPlan,
    EvidenceBundle,
    GenerationSpec,
    HARNESS_VERSION,
    HarnessIdentity,
    SkillManifest,
    SkillVerdict,
    StateDelta,
    TurnContext,
)
from .platform_safety import FinalGateDecision, PlatformSafetyGate, PreflightDecision
from .training import export_distillation_records, iter_distillation_records

__all__ = [
    "ActionPlan",
    "CallableModelAdapter",
    "EvidenceBundle",
    "FinalGateDecision",
    "export_distillation_records",
    "GenerationSpec",
    "HARNESS_VERSION",
    "HarnessIdentity",
    "HttpModelAdapter",
    "iter_distillation_records",
    "ModelAdapter",
    "OPSDModelAdapter",
    "PlatformSafetyGate",
    "PreflightDecision",
    "SFTModelAdapter",
    "SkillManifest",
    "SkillVerdict",
    "StateDelta",
    "TurnContext",
]
