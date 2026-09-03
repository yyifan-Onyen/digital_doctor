"""Hard readiness and output gates for treatment recommendations."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

from .risk import MoodAssessment


_ADVICE_PATTERNS = [
    r"\bexposure (exercise|task|practice|homework|plan)\b",
    r"\bresponse prevention\b",
    r"\b(exposure|erp) hierarchy\b",
    r"\b(delay|postpone) (washing|checking|cleaning|the ritual|your ritual|the compulsion)\b",
    r"\b(resist|stop|reduce|skip) (washing|checking|the ritual|your ritual|the compulsion|reassurance)\b",
    r"\bsit with (the )?(anxiety|uncertainty|urge|discomfort)\b",
    r"\bpractice (an|the|this|a)?\s*(exposure|response prevention|ritual prevention)\b",
    r"\btry (an|the|this|a)?\s*(exposure|delay|response prevention)\b",
    r"\bwould you be (open|willing) to (try|practice|delay|resist|stop|reduce)\b",
    r"\byour homework\b",
    r"\byou should (try|practice|delay|resist|stop|reduce)\b",
]
_ADVICE_RE = re.compile("|".join(_ADVICE_PATTERNS), re.IGNORECASE)
_RECOMMENDATION_CUE_RE = re.compile(
    r"\b(you should|you could|try to|consider|i recommend|i suggest|next time|"
    r"would you be (open|willing)|your (next step|homework|exercise)|the plan is to)\b",
    re.IGNORECASE,
)
_TREATMENT_ACTION_RE = re.compile(
    r"\b(exposure|erp|ritual|compulsion|washing|checking|cleaning|reassurance|"
    r"avoidance|uncertainty|anxiety|urge|medication|dose|mg|therapy|homework)\b",
    re.IGNORECASE,
)
_WITHOUT_RITUAL_RE = re.compile(
    r"\b(without|refrain from|do not|don'?t)\b.{0,50}\b(wash|check|clean|ritual|compulsion|reassurance)\w*\b",
    re.IGNORECASE,
)
_MEDICATION_ADVICE_RE = re.compile(
    r"\b(start|take|stop|skip|increase|decrease|double|halve|change)\b.{0,40}"
    r"\b(medication|medicine|dose|dosage|mg|prescription|antidepressant|ssri)\b",
    re.IGNORECASE,
)
_HIGH_RISK_ADVICE_RE = re.compile(
    r"\b(ingest|drink|swallow|taste|inhale|inject)\b.{0,50}"
    r"\b(chemical|cleaner|bleach|toxin|poison|bodily fluid|contaminated)\b|"
    r"\b(skip|avoid|delay|stop)\b.{0,50}\b(urgent care|medical care|doctor|prescribed medication)\b|"
    r"\b(exposure|practice)\b.{0,60}\b(weapon|fire|traffic|driving|height|allergen|overdose)\b",
    re.IGNORECASE,
)


@dataclass
class TreatmentReadiness:
    allowed: bool
    stage: str
    clinical_turns: int
    minimum_turns: int
    phase_prerequisites_met: bool
    missing_context: List[str]
    reasons: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "allowed": self.allowed,
            "stage": self.stage,
            "clinical_turns": self.clinical_turns,
            "minimum_turns": self.minimum_turns,
            "phase_prerequisites_met": self.phase_prerequisites_met,
            "missing_context": list(self.missing_context),
            "reasons": list(self.reasons),
        }


def assess_treatment_readiness(
    clinical_turns: int,
    formulation_fields: Mapping[str, object],
    mood: MoodAssessment,
    minimum_turns: int | None = None,
    phase_prerequisites_met: bool = True,
) -> TreatmentReadiness:
    configured_minimum = minimum_turns
    if configured_minimum is None:
        configured_minimum = int(os.getenv("TREATMENT_MIN_CONTEXT_TURNS", "3"))
    configured_minimum = max(2, configured_minimum)
    required = ("obsession", "trigger", "compulsion")
    missing: List[str] = []
    for name in required:
        value = formulation_fields.get(name)
        text = str(getattr(value, "value", value) or "").strip()
        if not text:
            missing.append(name)
    reasons: List[str] = []
    if clinical_turns < configured_minimum:
        reasons.append("insufficient_multi_turn_context")
    if missing:
        reasons.append("missing_case_formulation_context")
    if not phase_prerequisites_met:
        reasons.append("phase_prerequisites_incomplete")
    if not mood.treatment_allowed:
        reasons.append("mood_or_risk_not_stable")
    allowed = not reasons
    if mood.stop_conversation:
        stage = "stopped_for_safety"
    elif mood.stability == "unstable" or mood.risk_level == "high":
        stage = "paused_for_safety"
    elif not mood.treatment_allowed:
        stage = "paused_for_stability"
    elif clinical_turns < configured_minimum or missing or not phase_prerequisites_met:
        stage = "collecting_context"
    else:
        stage = "ready"
    return TreatmentReadiness(
        allowed=allowed,
        stage=stage,
        clinical_turns=clinical_turns,
        minimum_turns=configured_minimum,
        phase_prerequisites_met=phase_prerequisites_met,
        missing_context=missing,
        reasons=reasons,
    )


def contains_treatment_advice(text: str) -> bool:
    return bool(
        _ADVICE_RE.search(text)
        or _MEDICATION_ADVICE_RE.search(text)
        or _WITHOUT_RITUAL_RE.search(text)
        or (_RECOMMENDATION_CUE_RE.search(text) and _TREATMENT_ACTION_RE.search(text))
    )


def enforce_high_risk_treatment_limit(reply: str) -> Tuple[str, Dict[str, object]]:
    detected = bool(_HIGH_RISK_ADVICE_RE.search(reply) or _MEDICATION_ADVICE_RE.search(reply))
    if not detected:
        return reply, {"high_risk_advice_detected": False, "changed": False}
    return (
        "I can't safely guide that treatment or medication step in this setting. "
        "Please pause that plan and review it with a licensed clinician who can assess the risks directly.",
        {"high_risk_advice_detected": True, "changed": True},
    )


def treatment_policy_block(
    readiness: TreatmentReadiness,
    response_move: str = "treatment_step",
) -> str:
    if readiness.allowed:
        if response_move != "treatment_step":
            return (
                "Treatment readiness: READY, but this turn's selected dialogue move is "
                f"{response_move}. Readiness permits discussing a future treatment step; it does not itself "
                "authorize starting one. Do not prescribe, suggest, or invite a concrete exposure, "
                "response-prevention task, ritual delay, or homework on this turn. Follow the selected move only."
            )
        return (
            "Treatment readiness: READY. A single cautious ERP-consistent suggestion may be offered, "
            "but only because the selected dialogue move is treatment_step and only if it directly follows "
            "from the collected formulation."
        )
    missing = ", ".join(readiness.missing_context) or "none"
    phase_note = (
        "complete"
        if readiness.phase_prerequisites_met
        else "Assessment, Formulation, and ERP Buy-In are not all complete"
    )
    return (
        "Treatment readiness: NOT READY. Do not prescribe, suggest, or invite any exposure, response-prevention, "
        "ritual-delay, homework, medication, or other treatment action. Continue assessment and emotional support only. "
        f"Readiness stage: {readiness.stage}; missing context: {missing}; phase prerequisites: {phase_note}. "
        "Use the selected dialogue move and do not force a question when a brief acknowledgment or reflection is more natural."
    )


def _buffered_reply(readiness: TreatmentReadiness) -> str:
    if readiness.stage in {"paused_for_safety", "paused_for_stability", "stopped_for_safety"}:
        return (
            "You sound too unsettled for us to move into treatment steps safely right now, so I'm pausing that part of the conversation. "
            "Can you tell me whether you feel able to stay safe and connected to what is happening around you right now?"
        )
    question_by_field = {
        "obsession": "What unwanted thought or feared outcome is driving the distress most strongly?",
        "trigger": "What was happening just before the distress and urge began?",
        "compulsion": "What do you find yourself doing, avoiding, or asking for to get relief?",
    }
    question = next(
        (question_by_field[name] for name in readiness.missing_context if name in question_by_field),
        "What part of this pattern feels most important for me to understand before we discuss any next step?",
    )
    return (
        "Before we decide on any treatment step, I want to understand the pattern over a few turns and make sure the context is sufficient. "
        f"{question}"
    )


def enforce_treatment_buffer(
    reply: str,
    readiness: TreatmentReadiness,
    force_risk_hold: bool = True,
    treatment_step_selected: bool = True,
) -> Tuple[str, Dict[str, object]]:
    detected = contains_treatment_advice(reply)
    action_allowed = bool(readiness.allowed and treatment_step_selected)
    forced_risk_hold = bool(
        force_risk_hold and readiness.stage in {"paused_for_safety", "stopped_for_safety"}
    )
    changed = bool(forced_risk_hold or (detected and not action_allowed))
    if changed and readiness.allowed and not treatment_step_selected:
        final_reply = (
            "That makes sense. I want to stay with what you have just noticed rather than jumping ahead "
            "to an exercise. We can decide on the next step collaboratively and at a workable pace."
        )
    else:
        final_reply = _buffered_reply(readiness) if changed else reply
    return final_reply, {
        "advice_detected": detected,
        "advice_allowed": action_allowed,
        "readiness_allowed": readiness.allowed,
        "treatment_step_selected": treatment_step_selected,
        "changed": changed,
        "forced_risk_hold": forced_risk_hold,
        "stage": readiness.stage,
        "reasons": list(readiness.reasons),
    }
