"""Protocol implemented by every executable clinical skill."""

from __future__ import annotations

from typing import Dict, Protocol, Sequence, Tuple, runtime_checkable

from ..harness.contracts import (
    ActionPlan,
    HarnessIdentity,
    SkillManifest,
    StateDelta,
    TurnContext,
)


@runtime_checkable
class ClinicalSkill(Protocol):
    manifest: SkillManifest

    def identity(self, harness_version: str, model_adapter: str) -> HarnessIdentity:
        ...

    def create_tracker(self, milestones: Sequence[object], session_goal: str, event_writer):
        ...

    def assess_risk(self, context: TurnContext, trace):
        ...

    def observe(self, context: TurnContext, tracker: object, clinical: bool = True) -> StateDelta:
        ...

    def plan(self, context: TurnContext, tracker: object, route_decider) -> ActionPlan:
        ...

    def assess_readiness(self, clinical_turns: int, tracker: object, mood: object, minimum_turns):
        ...

    def phase_prerequisites_met(self, tracker: object) -> bool:
        ...

    def response_instructions(self, plan: ActionPlan) -> str:
        ...

    def treatment_policy(self, readiness: object, plan: ActionPlan) -> str:
        ...

    def helper_query_prompt(
        self,
        query: str,
        history: str,
        milestone_context: str,
        plan: ActionPlan,
    ) -> str:
        ...

    def helper_prompt(
        self,
        helper_query: str,
        history: str,
        milestone_context: str,
        plan: ActionPlan,
    ) -> str:
        ...

    def generate_chat(self, query: str, history: str, reason: str) -> str:
        ...

    def generate_analysis(self, **kwargs) -> str:
        ...

    def review_and_gate(
        self,
        query: str,
        draft_reply: str,
        history: str,
        tracker: object,
        readiness: object,
        mood: object,
        plan: ActionPlan,
        use_model_review: bool,
        trace,
    ) -> Tuple[str, object, Dict[str, object]]:
        ...


__all__ = ["ClinicalSkill"]
