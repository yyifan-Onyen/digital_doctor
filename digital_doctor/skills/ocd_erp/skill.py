"""Executable OCD/ERP clinical skill used by the Digital Doctor harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

from ...harness.contracts import (
    ActionPlan,
    HarnessIdentity,
    SkillManifest,
    StateDelta,
    TurnContext,
)
from ...tracking.milestones import MilestoneTracker
from .definition import (
    FORMULATION_FIELDS,
    PACKAGE_DIR,
    PHASE_SPECS,
    REQUIRED_PREREQUISITE_SLUGS,
    REQUIRED_TREATMENT_FIELDS,
    VALID_RESPONSE_MOVES,
)
from .planning import response_move_instructions
from .prompts import build_helper_prompt, build_helper_query_prompt
from .review import review_final_response
from .risk import assess_patient_state
from .tracking_policy import (
    CONTEXT_POLICY_LINES,
    apply_structured_phase_floors,
    build_formulation_update_prompt,
    build_phase_progress_prompt,
)
from .treatment import (
    TreatmentReadiness,
    assess_treatment_readiness,
    enforce_high_risk_treatment_limit,
    enforce_treatment_buffer,
    treatment_policy_block,
)


RiskAssessor = Callable[..., object]
ResponseReviewer = Callable[..., object]


def _bundle_checksum(root: Path) -> str:
    digest = hashlib.sha256()
    included = {
        "SKILL.md",
        "actions.json",
        "definition.py",
        "manifest.json",
        "phase_graph.json",
        "planning.py",
        "prompts.py",
        "review.py",
        "risk.py",
        "skill.py",
        "state_schema.json",
        "tracking_policy.py",
        "treatment.py",
    }
    for name in sorted(included):
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_manifest() -> SkillManifest:
    with (PACKAGE_DIR / "manifest.json").open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return SkillManifest(
        skill_id=str(raw["skill_id"]),
        name=str(raw["name"]),
        version=str(raw["version"]),
        description=str(raw["description"]),
        entry_point=str(raw["entry_point"]),
        capabilities=tuple(str(item) for item in raw.get("capabilities", [])),
        primary=bool(raw.get("primary", True)),
        bundle_checksum=_bundle_checksum(PACKAGE_DIR),
    )


class OcdErpSkill:
    """OCD state, policy, prompting, readiness, and response review."""

    def __init__(
        self,
        *,
        risk_assessor: Optional[RiskAssessor] = None,
        response_reviewer: Optional[ResponseReviewer] = None,
        analysis_generator=None,
        chat_generator=None,
    ) -> None:
        self.manifest = _load_manifest()
        self._risk_assessor = risk_assessor or assess_patient_state
        self._response_reviewer = response_reviewer or review_final_response
        self._analysis_generator = analysis_generator
        self._chat_generator = chat_generator

    def create_tracker(self, milestones: Sequence[object], session_goal: str, event_writer):
        return MilestoneTracker(
            milestones,
            session_goal=session_goal,
            event_writer=event_writer,
            phase_specs=PHASE_SPECS,
            formulation_fields=FORMULATION_FIELDS,
            formulation_prompt_builder=build_formulation_update_prompt,
            phase_prompt_builder=build_phase_progress_prompt,
            phase_floor_policy=apply_structured_phase_floors,
            context_policy_lines=CONTEXT_POLICY_LINES,
        )

    def identity(self, harness_version: str, model_adapter: str) -> HarnessIdentity:
        return HarnessIdentity(
            harness_version=harness_version,
            skill_id=self.manifest.skill_id,
            skill_version=self.manifest.version,
            skill_checksum=self.manifest.bundle_checksum,
            model_adapter=model_adapter,
        )

    def assess_risk(self, context: TurnContext, trace):
        return self._risk_assessor(context.query, context.history, trace=trace)

    def observe(
        self,
        context: TurnContext,
        tracker: MilestoneTracker,
        clinical: bool = True,
    ) -> StateDelta:
        if not clinical:
            metadata = {
                "formulation_updates": [],
                "formulation_filled_count": len(tracker.formulation.filled_fields()),
                "formulation_total_fields": len(tracker.formulation.fields),
                "skipped_for_chat": True,
            }
            return StateDelta(metadata=metadata)
        metadata = tracker.observe_user_turn(context.query, context.history)
        return StateDelta(
            updates=list(metadata.get("formulation_updates", [])),
            metadata=dict(metadata),
        )

    def plan(self, context: TurnContext, tracker: MilestoneTracker, route_decider) -> ActionPlan:
        route = route_decider(context.query, context.history, tracker.render_context())
        plan = ActionPlan.from_mapping(route)
        plan.allowed_actions = sorted(VALID_RESPONSE_MOVES)
        return plan

    def phase_prerequisites_met(self, tracker: MilestoneTracker) -> bool:
        required = [
            phase for phase in tracker.phases if phase.slug in REQUIRED_PREREQUISITE_SLUGS
        ]
        return bool(required) and all(
            tracker.state[phase.phase_id].status == "completed" for phase in required
        )

    def assess_readiness(
        self,
        clinical_turns: int,
        tracker: MilestoneTracker,
        mood: object,
        minimum_turns: Optional[int],
    ) -> TreatmentReadiness:
        return assess_treatment_readiness(
            clinical_turns=clinical_turns,
            formulation_fields=tracker.formulation.fields,
            mood=mood,
            minimum_turns=minimum_turns,
            phase_prerequisites_met=self.phase_prerequisites_met(tracker),
            required_fields=REQUIRED_TREATMENT_FIELDS,
        )

    def generate_chat(self, query: str, history: str, reason: str) -> str:
        if self._chat_generator is None:
            raise RuntimeError("The OCD/ERP skill has no configured chat generator")
        return str(self._chat_generator(query, history, reason))

    def generate_analysis(self, **kwargs) -> str:
        if self._analysis_generator is None:
            raise RuntimeError("The OCD/ERP skill has no configured analysis generator")
        return str(self._analysis_generator(**kwargs))

    @staticmethod
    def response_instructions(plan: ActionPlan) -> str:
        return response_move_instructions(plan.to_route_dict())

    @staticmethod
    def treatment_policy(readiness: TreatmentReadiness, plan: ActionPlan) -> str:
        return treatment_policy_block(readiness, plan.action)

    @staticmethod
    def helper_query_prompt(
        query: str,
        history: str,
        milestone_context: str,
        plan: ActionPlan,
    ) -> str:
        return build_helper_query_prompt(
            query,
            history,
            milestone_context,
            response_move=plan.action,
        )

    @staticmethod
    def helper_prompt(
        helper_query: str,
        history: str,
        milestone_context: str,
        plan: ActionPlan,
    ) -> str:
        return build_helper_prompt(
            helper_query,
            history,
            milestone_context,
            response_move=plan.action,
        )

    def review_and_gate(
        self,
        query: str,
        draft_reply: str,
        history: str,
        tracker: MilestoneTracker,
        readiness: TreatmentReadiness,
        mood: object,
        plan: ActionPlan,
        use_model_review: bool,
        trace,
    ) -> Tuple[str, Optional[Dict[str, object]], Dict[str, object]]:
        treatment_step_selected = plan.action == "treatment_step" and plan.authorized
        action_allowed = bool(readiness.allowed and treatment_step_selected)
        limited_reply, high_risk_before = enforce_high_risk_treatment_limit(draft_reply)
        buffered_reply, before_meta = enforce_treatment_buffer(
            limited_reply,
            readiness,
            treatment_step_selected=treatment_step_selected,
        )
        if not use_model_review:
            limited_final, high_risk_after = enforce_high_risk_treatment_limit(buffered_reply)
            final_reply, after_meta = enforce_treatment_buffer(
                limited_final,
                readiness,
                force_risk_hold=False,
                treatment_step_selected=treatment_step_selected,
            )
            return final_reply, None, {
                "high_risk_before_safety": high_risk_before,
                "before_safety": before_meta,
                "high_risk_final_check": high_risk_after,
                "final_check": after_meta,
            }

        verdict = self._response_reviewer(
            user_text=query,
            draft_reply=buffered_reply,
            history=history,
            formulation_context=tracker.formulation.render_context(),
            treatment_allowed=action_allowed,
            mood_assessment=mood.to_dict(),
            trace=trace,
        )
        limited_final, high_risk_after = enforce_high_risk_treatment_limit(verdict.final_reply)
        final_reply, after_meta = enforce_treatment_buffer(
            limited_final,
            readiness,
            force_risk_hold=False,
            treatment_step_selected=treatment_step_selected,
        )
        safety_meta = verdict.to_dict()
        high_risk_changed = bool(high_risk_before["changed"] or high_risk_after["changed"])
        treatment_buffer_changed = bool(before_meta["changed"] or after_meta["changed"])
        if treatment_buffer_changed or high_risk_changed:
            safety_meta["action"] = "revise"
            safety_meta["changed"] = True
            categories = list(safety_meta.get("categories", []))
            treatment_category = (
                "treatment_move_not_selected"
                if readiness.allowed and not treatment_step_selected
                else "treatment_not_ready"
            )
            if treatment_buffer_changed and treatment_category not in categories:
                categories.append(treatment_category)
            if high_risk_changed and "unsafe_treatment" not in categories:
                categories.append("unsafe_treatment")
            safety_meta["categories"] = categories
            safety_meta["rationale"] = (
                "Deterministic output guardrail removed unsafe or premature treatment guidance."
            )
        return final_reply, safety_meta, {
            "high_risk_before_safety": high_risk_before,
            "before_safety": before_meta,
            "high_risk_final_check": high_risk_after,
            "final_check": after_meta,
        }


__all__ = ["OcdErpSkill"]
