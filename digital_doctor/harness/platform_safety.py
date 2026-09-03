"""Non-bypassable safety decisions owned by the harness control plane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from .contracts import ActionPlan


@dataclass(frozen=True)
class PreflightDecision:
    stop: bool
    reason: str = ""
    response: str = ""


@dataclass(frozen=True)
class FinalGateDecision:
    reply: str
    changed: bool = False
    reason: str = "allow"


class PlatformSafetyGate:
    """Enforce stop state and action authorization independently of generation."""

    def __init__(self, stopped_response, critical_response) -> None:
        self._stopped_response = stopped_response
        self._critical_response = critical_response

    def preflight(self, already_stopped: bool, risk_assessment: Optional[object]) -> PreflightDecision:
        if already_stopped:
            return PreflightDecision(
                stop=True,
                reason="existing_safety_stop",
                response=self._stopped_response(),
            )
        if bool(getattr(risk_assessment, "stop_conversation", False)):
            return PreflightDecision(
                stop=True,
                reason="critical_risk",
                response=self._critical_response(),
            )
        return PreflightDecision(stop=False)

    def authorize(self, plan: ActionPlan, readiness: object) -> ActionPlan:
        allowed = plan.action in plan.allowed_actions if plan.allowed_actions else True
        reason = "skill_action_allowed" if allowed else "action_not_declared_by_skill"
        if plan.action == "treatment_step" and not bool(getattr(readiness, "allowed", False)):
            allowed = False
            reason = "treatment_readiness_denied"
        plan.authorized = allowed
        plan.authorization_reason = reason
        return plan

    def finalize(
        self,
        reply: str,
        risk_assessment: object,
        plan: ActionPlan,
        skill_gate_metadata: Optional[Mapping[str, object]] = None,
    ) -> FinalGateDecision:
        """Fail closed if a skill omits proof that an unauthorized action was removed."""
        if bool(getattr(risk_assessment, "stop_conversation", False)):
            return FinalGateDecision(
                reply=self._critical_response(),
                changed=True,
                reason="critical_risk_after_generation",
            )
        if not reply.strip():
            return FinalGateDecision(
                reply=(
                    "I can't safely provide a response from this system right now. "
                    "Please pause and contact a licensed clinician for support."
                ),
                changed=True,
                reason="empty_skill_output",
            )
        if not plan.authorized:
            metadata = dict(skill_gate_metadata or {})
            final_check = metadata.get("final_check", {})
            proven_safe = isinstance(final_check, dict) and not bool(
                final_check.get("advice_detected", True)
            )
            if not proven_safe:
                return FinalGateDecision(
                    reply=(
                        "Before we take any treatment step, I want to make sure there is enough "
                        "context and that the plan has been reviewed safely."
                    ),
                    changed=True,
                    reason="unauthorized_action_not_proven_removed",
                )
        return FinalGateDecision(reply=reply)


__all__ = ["FinalGateDecision", "PlatformSafetyGate", "PreflightDecision"]
