"""OCD/ERP state-extraction prompts and deterministic phase policies."""

from __future__ import annotations

from typing import Dict, List, Sequence, Set


CONTEXT_POLICY_LINES = (
    "Transition rule: address the smallest unmet current-phase criterion. If the latest patient evidence satisfies the current exit criteria, briefly consolidate that learning and bridge to the next phase shown above; otherwise remain in the current phase. Never announce phase labels to the patient or skip an unresolved phase.",
    "Constraint: avoid reassurance, stay ERP-consistent, and use at most one focused question or one concrete next step as allowed by the selected dialogue move and treatment-readiness policy.",
)


def build_formulation_update_prompt(
    formulation_block: str,
    history: str,
    user_text: str,
    field_names: Sequence[str],
) -> str:
    field_menu = "|".join(field_names)
    return f"""
You are updating a structured OCD case formulation from one patient turn.
Use only information that is explicit in the latest user message or the recent dialogue summary.

Return strict JSON with this schema:
{{
  "updates": [
    {{
      "field": "{field_menu}",
      "value": "short clinician note",
      "confidence": 0.0,
      "evidence": "short quote or paraphrase"
    }}
  ]
}}

Rules:
- Only include fields with new evidence.
- Keep each value short and concrete.
- Do not guess or infer hidden history.
- "obsession" is the recurring feared theme or meaning, such as contamination,
  illness, responsibility, harm, or taboo content. The patient does not need to
  use the word "obsession". If they explicitly describe an object as contaminated
  and feel driven to neutralize it, "contamination fear" is valid obsession evidence.
- "trigger" is the situation, object, thought, image, or sensation that activates
  the fear. "feared_consequence" is what the patient thinks may happen next.
- "compulsion" includes an explicitly reported urge or repeated action intended to
  neutralize distress, even when the patient has not performed it on this occasion.
- If a core field is still empty and the recent dialogue contains explicit evidence,
  include that update; do not leave a clearly stated trigger-fear-ritual loop blank.
- "wins" means reported progress.
- "stuck_points" means barriers, confusion, refusal, or repeated ritual pull.

Current formulation:
{formulation_block}

Recent dialogue summary:
{history if history else "(none)"}

Latest user message:
{user_text}
""".strip()


def build_phase_progress_prompt(
    session_goal: str,
    formulation_summary: str,
    phase_lines: str,
    user_text: str,
    assistant_text: str,
) -> str:
    return f"""
You are evaluating phase progress for one ERP therapy turn.
Use only the structured formulation, the phase definitions, and this turn's dialogue.

Return strict JSON with this schema:
{{
  "phases": [
    {{
      "id": 1,
      "status": "pending|active|completed|blocked|contraindicated",
      "confidence": 0.0,
      "evidence": "short quote or paraphrase",
      "blocked_reason": "",
      "contraindication_reason": ""
    }}
  ]
}}

Rules:
- Include every phase exactly once.
- The earliest unresolved phase is the planning priority.
- Do not skip to later phases unless earlier phases are already sufficiently established.
- Assessment can complete once the symptom theme, trigger, ritual/avoidance, distress,
  and functional impact are concrete enough to summarize; every formulation field is not required.
- Formulation can complete when a specific trigger-fear/meaning-ritual-maintenance loop
  is available to guide the next move; every optional field is not required.
- ERP Buy-In can complete when the patient demonstrates a tentative understanding of
  the learning rationale or willingness to try, even if anxiety remains high.
- Ordinary OCD anxiety or a high SUDS rating does not block phase progress when the
  patient remains coherent and engaged.
- Use "blocked" when the current phase is stalled.
- Use "contraindicated" when that phase should stop and human review or a different step is needed.
- Keep confidence in [0, 1].

Session goal:
{session_goal}

Structured formulation:
{formulation_summary}

Phases:
{phase_lines}

Current turn:
User: {user_text}
Assistant: {assistant_text}
""".strip()


def apply_structured_phase_floors(
    turn_idx: int,
    filled_fields: Set[str],
    parsed: Dict[int, Dict[str, object]],
    user_text: str,
    buy_in_evidence,
) -> List[Dict[str, object]]:
    overrides: List[Dict[str, object]] = []
    assessment_ready = (
        turn_idx >= 4
        and {"obsession", "trigger", "compulsion"}.issubset(filled_fields)
        and bool(filled_fields & {"avoidance", "feared_consequence", "stuck_points", "insight"})
    )
    formulation_ready = (
        turn_idx >= 8
        and {"obsession", "trigger", "feared_consequence", "compulsion"}.issubset(filled_fields)
        and bool(filled_fields & {"avoidance", "stuck_points", "insight"})
    )

    if assessment_ready:
        parsed[1] = {
            "status": "completed",
            "confidence": 1.0,
            "evidence": "Structured formulation contains a concrete obsession-trigger-compulsion pattern and impact context.",
            "blocked_reason": "",
            "contraindication_reason": "",
        }
        overrides.append({"phase_id": 1, "reason": "assessment_structured_floor"})
    if formulation_ready:
        parsed[1] = {
            "status": "completed",
            "confidence": 1.0,
            "evidence": "Assessment prerequisites remain satisfied.",
            "blocked_reason": "",
            "contraindication_reason": "",
        }
        parsed[2] = {
            "status": "completed",
            "confidence": 1.0,
            "evidence": "Structured formulation contains trigger, feared consequence, ritual, and a maintaining or impact factor.",
            "blocked_reason": "",
            "contraindication_reason": "",
        }
        overrides.extend(
            [
                {"phase_id": 1, "reason": "formulation_prerequisite_floor"},
                {"phase_id": 2, "reason": "formulation_structured_floor"},
            ]
        )
    if formulation_ready and buy_in_evidence.search(user_text):
        parsed[3] = {
            "status": "completed",
            "confidence": 1.0,
            "evidence": "Patient expressed tentative willingness or accurately summarized the short-term/long-term ERP rationale.",
            "blocked_reason": "",
            "contraindication_reason": "",
        }
        overrides.append({"phase_id": 3, "reason": "buy_in_evidence_floor"})
    return overrides


__all__ = [
    "CONTEXT_POLICY_LINES",
    "apply_structured_phase_floors",
    "build_formulation_update_prompt",
    "build_phase_progress_prompt",
]
