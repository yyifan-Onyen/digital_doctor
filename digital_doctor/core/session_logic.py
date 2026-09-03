from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..prompt import (
    DEFAULT_SYSTEM,
    FINAL_DRAFT_PROMPT_MILESTONE,
    FINAL_POLISH_PROMPT_MILESTONE,
    build_casual_chat_prompt,
)
from ..harness.contracts import GenerationSpec
from ..services.openai_client import call_model
from ..skills.ocd_erp.definition import (
    VALID_RESPONSE_DEPTHS as SKILL_RESPONSE_DEPTHS,
    VALID_RESPONSE_MOVES as SKILL_RESPONSE_MOVES,
)
from ..skills.ocd_erp.planning import (
    decide_route as decide_ocd_erp_route,
    fallback_response_move,
    response_move_instructions as ocd_erp_response_move_instructions,
)
from ..tracking.milestones import MilestoneTracker
from .text_utils import extract_final


TraceWriter = Callable[[str, Dict[str, object]], None]

VALID_RESPONSE_MOVES = set(SKILL_RESPONSE_MOVES)
VALID_RESPONSE_DEPTHS = set(SKILL_RESPONSE_DEPTHS)


def clip_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def looks_like_instruction_list(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return True
    if re.search(r"^\s*1\.\s", text):
        return True
    cues = [
        "start with",
        "ask a casual question",
        "share a brief",
        "if the conversation feels stuck",
        "step 1",
        "step 2",
    ]
    return any(cue in lowered for cue in cues)


def default_progress_update(tracker: MilestoneTracker, route: str) -> Dict[str, object]:
    covered = sum(1 for phase in tracker.phases if tracker.state[phase.phase_id].status == "completed")
    total = len(tracker.phases)
    next_target = tracker.next_target()
    return {
        "turn_idx": tracker.turn_idx,
        "route": route,
        "covered_now": [],
        "partial_now": [],
        "blocked_now": [],
        "contraindicated_now": [],
        "covered_count": covered,
        "total": total,
        "next_target": f"P{next_target.phase_id}: {next_target.title}" if next_target else "All phases completed",
        "current_phase": f"P{next_target.phase_id}: {next_target.title}" if next_target else "Completed",
        "top_scores": [],
        "status_changes": [],
        "transition": {
            "advanced": False,
            "from_phase_id": next_target.phase_id if next_target else None,
            "from_phase": next_target.title if next_target else "Completed",
            "to_phase_id": next_target.phase_id if next_target else None,
            "to_phase": next_target.title if next_target else "Completed",
        },
        "milestone_health": tracker.health(),
        "formulation_filled_count": len(tracker.formulation.filled_fields()),
        "formulation_total_fields": len(tracker.formulation.fields),
    }


def serialize_knowledge_hits(knowledge_hits: Sequence[object]) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for hit in knowledge_hits:
        node = getattr(hit, "node", None)
        if node is None:
            continue
        items.append(
            {
                "title": getattr(node, "title", ""),
                "parent_title": getattr(node, "parent_title", ""),
                "score": round(float(getattr(hit, "score", 0.0)), 4),
                "summary": getattr(node, "summary", ""),
                "clinical_uses": list(getattr(node, "clinical_uses", [])[:2]),
                "avoidances": list(getattr(node, "avoidances", [])[:2]),
                "canonical_topics": list(getattr(node, "canonical_topics", [])[:4]),
            }
        )
    return items


def build_turn_artifacts(
    route: str,
    history: str,
    milestone_context: str,
    milestone_snapshot_before: Dict[str, object],
    refs: Sequence[str],
    knowledge_hits: Sequence[object],
    helper_query: str = "",
    helper_answer: str = "",
    source_candidates: Optional[Dict[str, str]] = None,
    milestone_snapshot_after: Optional[Dict[str, object]] = None,
    pre_turn_observation: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    artifacts: Dict[str, object] = {
        "route": route,
        "history_excerpt": clip_text(history, limit=1500) if history else "",
        "milestone_context_before_turn": milestone_context,
        "milestone_snapshot_before_turn": milestone_snapshot_before,
        "transcript_refs": list(refs),
        "knowledge_hits": serialize_knowledge_hits(knowledge_hits),
        "helper_query": helper_query,
        "helper_answer": helper_answer,
    }
    if pre_turn_observation is not None:
        artifacts["pre_turn_observation"] = pre_turn_observation
    if source_candidates is not None:
        artifacts["source_candidates"] = source_candidates
    if milestone_snapshot_after is not None:
        artifacts["milestone_snapshot_after_turn"] = milestone_snapshot_after
    return artifacts


def generate_analysis_reply(
    query: str,
    history: str,
    history_block: str,
    milestone_context: str,
    milestone_block: str,
    refs_block: str,
    helper_block: str,
    knowledge_block: str,
    treatment_policy_block: str,
    response_move_block: str,
    candidate_name: str,
    trace: TraceWriter,
    model_adapter: Optional[object] = None,
) -> str:
    draft_prompt = FINAL_DRAFT_PROMPT_MILESTONE.format(
        system_msg=DEFAULT_SYSTEM,
        history_block=(history_block + "\n") if history_block else "",
        refs_block=refs_block,
        helper_block=helper_block,
        knowledge_block=knowledge_block,
        milestone_block=milestone_block + "\n",
        response_move_block=response_move_block + "\n",
        treatment_policy_block=treatment_policy_block + "\n",
        user_msg=query,
    )
    if model_adapter is None:
        draft_raw = call_model(draft_prompt, json_mode=False)
    else:
        draft_raw = model_adapter.generate(
            GenerationSpec(
                stage="draft",
                prompt=draft_prompt,
                json_mode=False,
                candidate_name=candidate_name,
            )
        )
    draft_text = extract_final(draft_raw)
    trace(
        "draft_generated",
        {
            "candidate_name": candidate_name,
            "used_refs": bool(refs_block.strip()),
            "used_knowledge": bool(knowledge_block.strip()),
            "draft_prompt": clip_text(draft_prompt),
            "draft_raw": clip_text(draft_raw),
            "draft_text": clip_text(draft_text),
        },
    )

    polish_prompt = FINAL_POLISH_PROMPT_MILESTONE.format(
        system_msg=DEFAULT_SYSTEM,
        history_block=history if history else "(session start)",
        refs_block=refs_block,
        knowledge_block=knowledge_block.strip() if knowledge_block else "(none)",
        milestone_block=milestone_context,
        response_move_block=response_move_block,
        treatment_policy_block=treatment_policy_block,
        draft_text=draft_text,
    )
    if model_adapter is None:
        final_raw = call_model(polish_prompt, json_mode=False)
    else:
        final_raw = model_adapter.generate(
            GenerationSpec(
                stage="polish",
                prompt=polish_prompt,
                json_mode=False,
                candidate_name=candidate_name,
            )
        )
    final_text = extract_final(final_raw).strip()
    if not final_text:
        final_text = draft_text.strip()
    trace(
        "polish_generated",
        {
            "candidate_name": candidate_name,
            "used_refs": bool(refs_block.strip()),
            "used_knowledge": bool(knowledge_block.strip()),
            "polish_prompt": clip_text(polish_prompt),
            "final_raw": clip_text(final_raw),
            "final_text": clip_text(final_text),
        },
    )
    return final_text


def summarize_dialogue_memory(
    existing_summary: str,
    turns: Sequence[Tuple[str, str]],
) -> Mapping[str, object]:
    """Condense an older memory chunk while carrying the prior summary forward."""
    dialogue = "\n".join(f"{role}: {text}" for role, text in turns)
    prompt = f"""
You maintain the durable long-term memory for an OCD clinical dialogue.
Merge the prior summary with the newly archived dialogue. Preserve only facts
actually discussed, including recurring themes, triggers, compulsions, prior
clinician explanations, agreed plans, reported outcomes, preferences, and safety
events. Do not diagnose, infer intent, or add facts. Keep old information that is
still relevant and update it when the new dialogue explicitly supersedes it.

Return strict JSON:
{{
  "summary": "compact but comprehensive longitudinal summary",
  "topics": ["short topic with relevant detail"],
  "reminders": ["prior plan or fact that may be useful to remind the patient about"]
}}

Prior long-term summary:
{existing_summary if existing_summary else "(none yet)"}

Newly archived dialogue:
{dialogue}
""".strip()
    raw = call_model(prompt, json_mode=True)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {"summary": raw.strip(), "topics": [], "reminders": []}
    topics = obj.get("topics", [])
    reminders = obj.get("reminders", [])
    return {
        "summary": str(obj.get("summary", "")).strip(),
        "topics": topics if isinstance(topics, list) else [],
        "reminders": reminders if isinstance(reminders, list) else [],
    }


def _fallback_response_move(query: str, mode: str) -> str:
    return fallback_response_move(query, mode)


def response_move_instructions(route: Mapping[str, str]) -> str:
    return ocd_erp_response_move_instructions(route)


def decide_route(
    query: str,
    history: str,
    trace: TraceWriter,
    milestone_context: str = "",
) -> Dict[str, str]:
    return decide_ocd_erp_route(
        query,
        history,
        trace,
        milestone_context,
        model_call=call_model,
    )


def generate_chat_reply(
    query: str,
    history: str,
    route_reason: str,
    trace: TraceWriter,
    model_adapter: Optional[object] = None,
) -> str:
    casual_prompt = build_casual_chat_prompt(query, history)
    if model_adapter is None:
        raw = call_model(casual_prompt, json_mode=False)
    else:
        raw = model_adapter.generate(
            GenerationSpec(stage="chat", prompt=casual_prompt, json_mode=False)
        )
    gen = extract_final(raw).strip()
    used_fallback = False
    if looks_like_instruction_list(gen):
        fallback_prompt = (
            "Rewrite this into a natural doctor-to-patient reply.\n"
            "Constraints: 1-3 sentences; supportive but natural; no bullet points; no meta advice.\n\n"
            f"Patient message:\n{query}\n\n"
            f"Draft to rewrite:\n{gen}"
        )
        if model_adapter is None:
            fallback_raw = call_model(fallback_prompt, json_mode=False)
        else:
            fallback_raw = model_adapter.generate(
                GenerationSpec(stage="chat_rewrite", prompt=fallback_prompt, json_mode=False)
            )
        gen = extract_final(fallback_raw).strip()
        used_fallback = True
    if not gen:
        gen = "Good to see you. How are you feeling right now?"
    trace(
        "chat_response_generated",
        {
            "casual_prompt": clip_text(casual_prompt),
            "final_text": clip_text(gen),
            "route_reason": route_reason,
            "used_fallback": used_fallback,
        },
    )
    return gen
