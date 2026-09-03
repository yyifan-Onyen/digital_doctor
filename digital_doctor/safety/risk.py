"""Pre-generation patient mood, stability, and acute-risk assessment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..services.openai_client import call_model


TraceWriter = Callable[[str, Dict[str, object]], None]
VALID_STABILITY = {"stable", "strained", "unstable", "critical"}
VALID_RISK = {"low", "moderate", "high", "critical"}

_CRITICAL_RE = re.compile(
    r"\b(i want to (die|kill myself|end my life|hurt myself)|"
    r"i am going to (kill myself|end my life|hurt myself)|"
    r"i have a plan to (kill myself|end my life|hurt myself)|"
    r"i intend to (kill myself|hurt myself|kill|hurt)\b|"
    r"i have a plan to (kill|hurt) (someone|him|her|them)\b|"
    r"i am going to (kill|hurt) (someone|him|her|them)\b|"
    r"i will (kill myself|end my life|hurt myself)\b|"
    r"about to (kill myself|hurt myself)|can'?t keep myself safe)\b",
    re.IGNORECASE,
)
_UNSTABLE_RE = re.compile(
    r"\b(panic attack|out of control|can'?t think straight|cannot think straight|"
    r"can'?t cope|cannot cope|too overwhelmed to (continue|talk|participate)|dissociat\w*|haven'?t slept for (days|[2-9] days)|"
    r"have not slept for (days|[2-9] days)|hearing voices|seeing things)\b",
    re.IGNORECASE,
)
_STRAINED_RE = re.compile(
    r"\b(hard to think|can'?t focus|cannot focus|struggling to stay present|"
    r"need to stop|distress is too much|barely holding it together)\b",
    re.IGNORECASE,
)
_ORDINARY_DISTRESS_RE = re.compile(
    r"\b(anxious|anxiety|sad|upset|angry|scared|distressed|stress(?:ed)?|"
    r"grossed out|disgusted|exhausted|overwhelmed)\b",
    re.IGNORECASE,
)
_OBSESSIONAL_RE = re.compile(
    r"\b(intrusive|unwanted|ego-dystonic|what if|afraid i might|scared i might|"
    r"fear i might|don'?t want to)\b",
    re.IGNORECASE,
)


@dataclass
class MoodAssessment:
    mood: str = "unknown"
    stability: str = "strained"
    risk_level: str = "moderate"
    categories: List[str] = field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    treatment_allowed: bool = False
    notify_human: bool = False
    stop_conversation: bool = False
    source: str = "model"

    def to_dict(self) -> Dict[str, object]:
        return {
            "mood": self.mood,
            "stability": self.stability,
            "risk_level": self.risk_level,
            "categories": list(self.categories),
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "treatment_allowed": self.treatment_allowed,
            "notify_human": self.notify_human,
            "stop_conversation": self.stop_conversation,
            "source": self.source,
        }


def _prompt(user_text: str, history: str) -> str:
    return f"""
You are the pre-response risk assessor for an OCD/ERP clinical support system.
Assess the patient's CURRENT mood, emotional stability, and immediate risk before
the dialogue system is allowed to discuss treatment.

Return strict JSON:
{{
  "mood": "calm|anxious|sad|angry|overwhelmed|numb|activated|unknown",
  "stability": "stable|strained|unstable|critical",
  "risk_level": "low|moderate|high|critical",
  "categories": ["self_harm", "harm_to_others", "psychosis_or_mania", "severe_distress", "safeguarding", "other"],
  "confidence": 0.0,
  "rationale": "one short evidence-based sentence"
}}

Rules:
- critical: genuine imminent intent/plan/urge, inability to stay safe, or immediate danger.
- unstable/high: severe emotional dysregulation, acute psychosis/mania, inability to
  engage safely, or marked distress that makes treatment suggestions inappropriate.
- strained/moderate: clearly distressed but still oriented, coherent, and able to
  collaborate with support. Treatment may proceed cautiously if all other readiness
  and phase gates pass.
- stable/low: sufficiently oriented and able to collaborate. The mood may still be
  anxious, distressed, disgusted, or high on SUDS.
- Risk/stability measure acute safety and capacity to engage, not the subjective
  intensity of ordinary OCD anxiety. High anxiety, a strong ritual urge, crying, or
  a high SUDS rating alone must NOT be classified as strained/unstable when the
  patient remains coherent, oriented, and able to participate. ERP often occurs
  while anxiety is high.
- Unwanted, feared, ego-dystonic harm/taboo intrusive thoughts are OCD symptoms,
  not intent. Do not label them critical unless desire, plan, or intent is present.
- Do not infer facts not in the dialogue.

Dialogue memory:
{history if history else "(session start)"}

Latest patient message:
{user_text}
""".strip()


def _heuristic(user_text: str, reason: str) -> MoodAssessment:
    critical = bool(_CRITICAL_RE.search(user_text)) and not (
        _OBSESSIONAL_RE.search(user_text) and not re.search(r"\b(plan|intend|want to|going to|will)\b", user_text, re.I)
    )
    if critical:
        harm_category = (
            "self_harm"
            if re.search(r"\b(myself|my life|die)\b", user_text, re.I)
            else "harm_to_others"
        )
        return MoodAssessment(
            mood="overwhelmed",
            stability="critical",
            risk_level="critical",
            categories=[harm_category],
            confidence=0.8,
            rationale=f"Explicit immediate-safety language detected ({reason}).",
            treatment_allowed=False,
            notify_human=True,
            stop_conversation=True,
            source="heuristic",
        )
    if _UNSTABLE_RE.search(user_text):
        return MoodAssessment(
            mood="overwhelmed",
            stability="unstable",
            risk_level="high",
            categories=["severe_distress"],
            confidence=0.65,
            rationale=f"Marked instability language detected ({reason}).",
            treatment_allowed=False,
            notify_human=True,
            source="heuristic",
        )
    if _STRAINED_RE.search(user_text):
        return MoodAssessment(
            mood="anxious",
            stability="strained",
            risk_level="moderate",
            categories=["severe_distress"] if "overwhelmed" in user_text.lower() else [],
            confidence=0.45,
            rationale=f"Distress language detected without an acute-risk cue ({reason}).",
            treatment_allowed=True,
            source="heuristic",
        )
    if _ORDINARY_DISTRESS_RE.search(user_text):
        return MoodAssessment(
            mood="anxious",
            stability="stable",
            risk_level="low",
            confidence=0.45,
            rationale=f"Distress is present without evidence of impaired engagement or acute risk ({reason}).",
            treatment_allowed=True,
            source="heuristic",
        )
    return MoodAssessment(
        mood="unknown",
        stability="stable",
        risk_level="low",
        confidence=0.3,
        rationale=f"No explicit instability or acute-risk cue detected ({reason}).",
        treatment_allowed=True,
        source="heuristic",
    )


def _normalize(obj: Dict[str, object]) -> MoodAssessment:
    stability = str(obj.get("stability", "strained")).strip().lower()
    if stability not in VALID_STABILITY:
        stability = "strained"
    risk = str(obj.get("risk_level", "moderate")).strip().lower()
    if risk not in VALID_RISK:
        risk = "moderate"
    order = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
    required_risk = {"stable": "low", "strained": "moderate", "unstable": "high", "critical": "critical"}[stability]
    if order[risk] < order[required_risk]:
        risk = required_risk
    try:
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    categories = obj.get("categories", [])
    if not isinstance(categories, list):
        categories = []
    return MoodAssessment(
        mood=str(obj.get("mood", "unknown")).strip().lower()[:40] or "unknown",
        stability=stability,
        risk_level=risk,
        categories=[str(item) for item in categories][:8],
        confidence=confidence,
        rationale=str(obj.get("rationale", "")).strip()[:280],
        treatment_allowed=(
            stability in {"stable", "strained"}
            and risk in {"low", "moderate"}
        ),
        notify_human=risk in {"high", "critical"},
        stop_conversation=risk == "critical" or stability == "critical",
        source="model",
    )


def assess_patient_state(
    user_text: str,
    history: str = "",
    trace: Optional[TraceWriter] = None,
) -> MoodAssessment:
    prompt = _prompt(user_text, history)
    try:
        raw = call_model(prompt, json_mode=True)
        assessment = _normalize(json.loads(raw))
    except Exception:
        raw = ""
        assessment = _heuristic(user_text, "risk model unavailable")
    if trace is not None:
        trace("mood_risk_assessment", {**assessment.to_dict(), "assessment_raw": raw[:1000]})
    return assessment


def critical_risk_response() -> str:
    return (
        "I'm concerned that you may not be safe right now, so I'm stopping the treatment discussion and alerting the clinical team. "
        "If you could act on these thoughts or are in immediate danger, call your local emergency number now; in the U.S. or Canada, call or text 988. "
        "Please stay with another person if you can while urgent human support is being arranged."
    )


def stopped_conversation_response() -> str:
    return (
        "This treatment conversation is paused because of the safety concern already identified. "
        "Please use the urgent support options provided and wait for a clinician or emergency professional to take over."
    )
