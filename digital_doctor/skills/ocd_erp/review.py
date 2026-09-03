"""Final-layer safety review.

This runs as the last step before a reply is returned to the patient. It looks
at the drafted reply together with the turn context and decides one of four
actions:

- ``allow``    -> return the draft unchanged
- ``revise``   -> rewrite the draft to remove a safety problem (e.g. it slipped
                  in reassurance, or coached a genuinely unsafe exposure), then
                  return the rewritten reply
- ``crisis``   -> replace the draft with a crisis-response message and flag for
                  human / emergency handoff
- ``escalate`` -> the case is out of scope for an automated ERP coach; return a
                  care-handoff message and flag for human review

OCD-specific nuance is baked into the prompt: ego-dystonic harm / taboo
obsessions are a normal OCD symptom and must NOT be treated as genuine intent,
and the coach must never provide reassurance (that feeds the compulsion).

The model call is the module-level ``call_model`` import so it can be mocked in
tests. If the review model is unavailable or returns unparseable output, a
conservative keyword backstop still catches explicit crisis language.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ...services.openai_client import call_model
from ...core.text_utils import extract_final
from .treatment import contains_treatment_advice


TraceWriter = Callable[[str, Dict[str, object]], None]
ModelCall = Callable[..., str]

VALID_ACTIONS = {"allow", "revise", "crisis", "escalate"}
VALID_RISK = {"none", "low", "elevated", "crisis"}
VALID_CATEGORIES = {
    "self_harm",
    "harm_to_others_intent",
    "abuse_or_safeguarding",
    "acute_psychiatric",
    "out_of_scope",
    "reassurance_given",
    "unsafe_exposure",
    "medical_or_medication_advice",
    "treatment_not_ready",
    "emotional_instability",
    "judgmental_or_shaming",
    "other",
}
_ACTIONABLE_REVISION_CATEGORIES = {
    "reassurance_given",
    "unsafe_exposure",
    "medical_or_medication_advice",
    "judgmental_or_shaming",
}

# Explicit crisis cues used ONLY as a fallback when the review model is
# unavailable / unparseable. Kept conservative to avoid pathologizing the
# ego-dystonic suicidal/harm *obsessions* that are common in OCD.
_CRISIS_PATTERNS = [
    r"\bkill myself\b",
    r"\bkilling myself\b",
    r"\bend my life\b",
    r"\bending my life\b",
    r"\btake my (own )?life\b",
    r"\bwant to die\b",
    r"\bi (don'?t|do not) want to (be alive|live)\b",
    r"\bcommit suicide\b",
    r"\bsuicidal\b",
    r"\bhurt myself\b",
    r"\bharm myself\b",
    r"\bcut myself\b",
    r"\boverdose\b",
]
_CRISIS_RE = re.compile("|".join(_CRISIS_PATTERNS), re.IGNORECASE)
_OBSESSIONAL_CONTEXT_RE = re.compile(
    r"\b(intrusive|unwanted|ego-dystonic|what if|afraid i might|scared i might|"
    r"fear i might|don'?t want to|do not want to)\b",
    re.IGNORECASE,
)
_INTENT_CONTEXT_RE = re.compile(
    r"\b(i want to|i intend to|i am going to|i will|i have a plan|about to|"
    r"can'?t keep myself safe|cannot keep myself safe)\b",
    re.IGNORECASE,
)

DEFAULT_CRISIS_RESOURCES = (
    "If you are in immediate danger, please call your local emergency number now "
    "(for example 911 in the US). In the US you can call or text 988 to reach the "
    "Suicide & Crisis Lifeline, available 24/7. If you are elsewhere, your local "
    "emergency services or a crisis line in your country can help right now."
)


def _crisis_resources() -> str:
    return os.getenv("SAFETY_CRISIS_RESOURCES", DEFAULT_CRISIS_RESOURCES).strip()


def _crisis_message() -> str:
    return (
        "I'm really glad you told me, and I'm concerned about your safety right now. "
        "This is more than I can safely help with on my own, and you deserve support "
        "from a person who can be with you.\n\n"
        f"{_crisis_resources()}\n\n"
        "You don't have to handle this alone — reaching out to one of these right now "
        "is the most important next step."
    )


def _escalation_message() -> str:
    return (
        "Thank you for trusting me with this. What you're describing is important, and "
        "it goes beyond what I should try to handle as an automated coach. I'd strongly "
        "encourage you to bring this to a licensed clinician who can give you the "
        "attention it deserves. If it ever feels urgent or unsafe, please contact local "
        "emergency services or a crisis line right away."
    )


@dataclass
class SafetyVerdict:
    action: str = "allow"
    risk_level: str = "none"
    categories: List[str] = field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    escalated: bool = False
    changed: bool = False
    final_reply: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "action": self.action,
            "risk_level": self.risk_level,
            "categories": list(self.categories),
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "escalated": self.escalated,
            "changed": self.changed,
        }


def _build_review_prompt(
    user_text: str,
    history: str,
    formulation_context: str,
    draft_reply: str,
    treatment_allowed: bool,
    mood_assessment: Optional[Dict[str, object]],
) -> str:
    return f"""
You are a clinical safety reviewer for an automated OCD/ERP (Exposure and Response
Prevention) coaching assistant. You are the LAST check before a drafted reply is
sent to the patient. Judge the patient's latest message, the context, and the
DRAFT REPLY, then decide a single action.

Return strict JSON with this schema:
{{
  "risk_level": "none|low|elevated|crisis",
  "categories": ["self_harm", "harm_to_others_intent", "abuse_or_safeguarding",
                 "acute_psychiatric", "out_of_scope", "reassurance_given",
                 "unsafe_exposure", "medical_or_medication_advice",
                 "treatment_not_ready", "emotional_instability",
                 "judgmental_or_shaming", "other"],
  "action": "allow|revise|crisis|escalate",
  "confidence": 0.0,
  "rationale": "one short sentence",
  "revision_instruction": "if action is revise, say exactly what to fix"
}}

Decision rules:
- "crisis": the patient expresses genuine intent, plan, or strong urge for suicide,
  self-harm, or harming another person; or is in immediate danger. Replace the reply.
- "escalate": out of scope for an automated ERP coach and needs a human clinician
  (e.g. acute psychosis/mania, severe depression, active abuse being disclosed,
  a minor without oversight, a serious medical/medication question).
- "revise": the DRAFT itself is unsafe or anti-therapeutic — it provides
  reassurance, answers an "am I sure / did I do it right" question, coaches a
  genuinely dangerous exposure (real physical harm, ingesting harmful things,
  skipping medically necessary care, stopping prescribed medication), gives
  specific medical/medication advice, or is judgmental/shaming.
- If TREATMENT ALLOWED is false, any exposure, response-prevention, ritual-delay,
  homework, medication, or other treatment recommendation must be "revise" and
  removed. The reply may support the patient and ask one assessment question.
- If mood stability is unstable/critical, do not allow treatment suggestions.
- Anxiety, distress, disgust, strong urges, or a high SUDS rating do not by
  themselves justify revising an otherwise safe reply. Do not use
  "emotional_instability" unless the supplied mood stability is actually
  unstable or critical.
- Review clinical safety, not stylistic preference. If the draft is safe but you
  would simply word it differently, choose "allow". The category "other" alone
  is not a sufficient reason to revise.
- "allow": none of the above; the draft is safe and ERP-consistent.

CRITICAL OCD nuance — do not get this wrong:
- Ego-dystonic harm, sexual, or taboo OBSESSIONS (unwanted, distressing intrusive
  thoughts the patient does NOT want to act on, often framed as "what if I..." or
  "I'm scared I might...") are ordinary OCD symptoms. These are NOT crisis or
  intent. Treating them as intent is clinically harmful. Only flag crisis/harm
  when there is genuine desire, plan, or intent — not a feared, unwanted thought.
- The coach should NOT reassure. If the draft reassures or certifies safety in
  response to an obsession, choose "revise", not "allow".

Recent dialogue:
{history if history else "(session start)"}

Structured case formulation:
{formulation_context if formulation_context else "(none)"}

Pre-generation mood/risk assessment:
{json.dumps(mood_assessment or {}, ensure_ascii=False)}

TREATMENT ALLOWED:
{str(treatment_allowed).lower()}

Patient's latest message:
{user_text}

DRAFT REPLY under review:
{draft_reply}
""".strip()


def _revise_reply(
    draft_reply: str,
    instruction: str,
    user_text: str,
    model_call: ModelCall = call_model,
) -> str:
    prompt = (
        "Rewrite the following ERP coaching reply to fix a safety/therapeutic problem, "
        "keeping the same warm, professional doctor-to-patient voice.\n"
        "Rules: stay ERP-consistent; do NOT provide reassurance or certainty about a "
        "feared outcome; do NOT coach anything physically dangerous or medical; keep it "
        "2-5 sentences; return only the rewritten reply.\n\n"
        f"Problem to fix: {instruction if instruction else 'Remove reassurance / unsafe guidance.'}\n\n"
        f"Patient message:\n{user_text}\n\n"
        f"Reply to rewrite:\n{draft_reply}"
    )
    try:
        revised = extract_final(model_call(prompt, json_mode=False)).strip()
    except Exception:
        return _escalation_message()
    return revised or _escalation_message()


def _fallback_verdict(
    user_text: str,
    draft_reply: str,
    reason: str,
    treatment_allowed: bool,
) -> SafetyVerdict:
    """Fail closed on crisis language or unevaluated treatment advice."""
    combined = f"{user_text}\n{draft_reply}"
    obsessional_without_intent = bool(_OBSESSIONAL_CONTEXT_RE.search(user_text)) and not bool(
        _INTENT_CONTEXT_RE.search(user_text)
    )
    if _CRISIS_RE.search(combined) and not obsessional_without_intent:
        return SafetyVerdict(
            action="crisis",
            risk_level="crisis",
            categories=["self_harm"],
            confidence=0.5,
            rationale=f"Keyword crisis backstop ({reason}).",
            escalated=True,
            changed=True,
            final_reply=_crisis_message(),
        )
    if contains_treatment_advice(draft_reply):
        treatment_category = "treatment_advice" if treatment_allowed else "treatment_not_ready"
        return SafetyVerdict(
            action="escalate",
            risk_level="elevated",
            categories=["safety_review_unavailable", treatment_category],
            confidence=0.5,
            rationale=f"Treatment advice was withheld because safety review was unavailable ({reason}).",
            escalated=True,
            changed=True,
            final_reply=_escalation_message(),
        )
    return SafetyVerdict(
        action="allow",
        risk_level="none",
        categories=[],
        confidence=0.0,
        rationale=f"Safety review unavailable ({reason}); no crisis keywords found.",
        final_reply=draft_reply,
    )


def _normalize(obj: Dict[str, object]) -> Dict[str, object]:
    risk = str(obj.get("risk_level", "none")).strip().lower()
    if risk not in VALID_RISK:
        risk = "none"
    action = str(obj.get("action", "")).strip().lower()
    if action not in VALID_ACTIONS:
        # Derive a safe action from the risk level when the model omits/garbles it.
        action = {"crisis": "crisis", "elevated": "escalate"}.get(risk, "allow")
    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    categories = obj.get("categories", [])
    if not isinstance(categories, list):
        categories = []
    normalized_categories = [
        str(c).strip()
        for c in categories
        if str(c).strip() in VALID_CATEGORIES
    ][:8]
    # Some small models copy the category menu from the requested schema instead
    # of selecting applicable items. Treat that mutually incompatible grab-bag
    # as invalid rather than allowing it to trigger an arbitrary rewrite.
    if len(normalized_categories) > 4:
        normalized_categories = []
    return {
        "risk_level": risk,
        "action": action,
        "confidence": confidence,
        "categories": normalized_categories,
        "rationale": str(obj.get("rationale", "")).strip()[:280],
        "revision_instruction": str(obj.get("revision_instruction", "")).strip(),
    }


def _revision_is_actionable(
    parsed: Dict[str, object],
    draft_reply: str,
    treatment_allowed: bool,
    mood_assessment: Optional[Dict[str, object]],
) -> bool:
    categories = {str(item) for item in parsed.get("categories", [])}
    if categories & _ACTIONABLE_REVISION_CATEGORIES:
        return True
    if (
        "treatment_not_ready" in categories
        and not treatment_allowed
        and contains_treatment_advice(draft_reply)
    ):
        return True
    stability = str((mood_assessment or {}).get("stability", "")).strip().lower()
    if (
        "emotional_instability" in categories
        and stability in {"unstable", "critical"}
        and contains_treatment_advice(draft_reply)
    ):
        return True
    return False


def review_final_response(
    user_text: str,
    draft_reply: str,
    history: str = "",
    formulation_context: str = "",
    treatment_allowed: bool = True,
    mood_assessment: Optional[Dict[str, object]] = None,
    trace: Optional[TraceWriter] = None,
    *,
    model_call: ModelCall = call_model,
) -> SafetyVerdict:
    """Review the drafted reply and return the final, safe reply plus metadata."""
    prompt = _build_review_prompt(
        user_text,
        history,
        formulation_context,
        draft_reply,
        treatment_allowed,
        mood_assessment,
    )
    try:
        raw = model_call(prompt, json_mode=True)
        parsed = _normalize(json.loads(raw))
    except (json.JSONDecodeError, TypeError) as exc:
        verdict = _fallback_verdict(
            user_text,
            draft_reply,
            f"parse:{type(exc).__name__}",
            treatment_allowed,
        )
        _emit(trace, verdict, prompt, "")
        return verdict
    except Exception as exc:  # model/transport failure
        verdict = _fallback_verdict(
            user_text,
            draft_reply,
            f"model:{type(exc).__name__}",
            treatment_allowed,
        )
        _emit(trace, verdict, prompt, "")
        return verdict

    action = parsed["action"]
    if action == "revise" and not _revision_is_actionable(
        parsed,
        draft_reply,
        treatment_allowed,
        mood_assessment,
    ):
        action = "allow"
        parsed["categories"] = []
        parsed["rationale"] = (
            "Non-actionable safety rewrite was ignored; the draft contained no explicitly categorized safety defect."
        )
    verdict = SafetyVerdict(
        action=action,
        risk_level=parsed["risk_level"],
        categories=parsed["categories"],
        confidence=parsed["confidence"],
        rationale=parsed["rationale"],
    )

    if action == "crisis":
        verdict.final_reply = _crisis_message()
        verdict.escalated = True
        verdict.changed = True
    elif action == "escalate":
        verdict.final_reply = _escalation_message()
        verdict.escalated = True
        verdict.changed = True
    elif action == "revise":
        revised = _revise_reply(
            draft_reply,
            parsed["revision_instruction"],
            user_text,
            model_call=model_call,
        )
        verdict.final_reply = revised
        verdict.changed = revised.strip() != draft_reply.strip()
    else:  # allow
        verdict.final_reply = draft_reply

    _emit(trace, verdict, prompt, raw)
    return verdict


def _emit(trace: Optional[TraceWriter], verdict: SafetyVerdict, prompt: str, raw: str) -> None:
    if trace is None:
        return
    trace(
        "safety_review",
        {
            "action": verdict.action,
            "risk_level": verdict.risk_level,
            "categories": verdict.categories,
            "confidence": round(verdict.confidence, 4),
            "rationale": verdict.rationale,
            "escalated": verdict.escalated,
            "changed": verdict.changed,
            "review_raw": raw[:1000],
        },
    )
