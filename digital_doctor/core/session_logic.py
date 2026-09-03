from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..prompt import (
    DEFAULT_SYSTEM,
    FINAL_DRAFT_PROMPT_MILESTONE,
    FINAL_POLISH_PROMPT_MILESTONE,
    TURN_MODE_PROMPT_MILESTONE,
    build_casual_chat_prompt,
)
from ..services.openai_client import call_model
from ..tracking.milestones import MilestoneTracker
from .text_utils import extract_final


TraceWriter = Callable[[str, Dict[str, object]], None]

VALID_RESPONSE_MOVES = {
    "casual",
    "acknowledge",
    "reflect",
    "clarify",
    "assess",
    "formulate",
    "psychoeducation",
    "build_buy_in",
    "treatment_step",
}
VALID_RESPONSE_DEPTHS = {"brief", "standard", "structured"}
_SHORT_BACKCHANNEL_RE = re.compile(
    r"^\s*(?:yeah|yep|yes|mm+|mhm+|mm-hm|uh-huh|right|okay|ok|no|and|thanks?|thank you|dev)"
    r"[\s.!?,…-]*$",
    re.IGNORECASE,
)
_EXPLANATION_REQUEST_RE = re.compile(
    r"\b(?:why|how come|what do you mean|can'?t remember|cannot remember|don'?t remember|"
    r"do not remember|remind me|what can i expect)\b",
    re.IGNORECASE,
)
_MEMORY_EXPLANATION_RE = re.compile(
    r"\b(?:can'?t remember|cannot remember|don'?t remember|do not remember|"
    r"not\s+(?:\w+\s+){0,3}remembering|remind me|what did we (?:say|discuss|decide|talk about))\b",
    re.IGNORECASE,
)
_LOW_INFORMATION_RE = re.compile(r"^\s*[a-z][a-z'-]{0,11}[.!?…]*\s*$", re.IGNORECASE)
_CLINICAL_HISTORY_RE = re.compile(
    r"\b(?:ocd|anxiety|distress|trigger|urge|wash|ritual|compulsion|exposure|"
    r"contaminat|avoid|therap|fear)\w*\b",
    re.IGNORECASE,
)


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
    draft_raw = call_model(draft_prompt, json_mode=False)
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
    final_raw = call_model(polish_prompt, json_mode=False)
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
    if mode == "chat":
        return "casual"
    if _SHORT_BACKCHANNEL_RE.match(query):
        return "acknowledge"
    if _EXPLANATION_REQUEST_RE.search(query):
        return "psychoeducation"
    if "?" in query:
        return "clarify"
    return "assess"


def response_move_instructions(route: Mapping[str, str]) -> str:
    move = str(route.get("response_move", "assess"))
    depth = str(route.get("depth", "standard"))
    guidance = {
        "casual": "Reply as an ordinary warm conversational partner; do not introduce clinical structure.",
        "acknowledge": "Treat the short backchannel as continuation, not new clinical content. Briefly acknowledge it, then continue the unfinished assessment, formulation, or buy-in thread from the immediately preceding dialogue. Do not restart with generic support; one focused question is allowed when it naturally continues that thread.",
        "reflect": "Reflect the patient's specific experience or meaning without teaching, summarizing the whole case, or rushing onward.",
        "clarify": "Resolve one ambiguity or answer the patient's narrow question directly; ask at most one clarification question.",
        "assess": "Gather one clinically useful missing detail with a natural transition and at most one focused question.",
        "formulate": "Collaboratively connect the relevant trigger, fear, ritual, relief, avoidance, or impairment into a concise shared pattern.",
        "psychoeducation": "Give the smallest useful mechanism explanation that answers the patient's need, then pause or ask one focused check-in if useful.",
        "build_buy_in": "Build an ERP rationale, explore the patient's concern, or invite willingness without pressuring them or jumping ahead.",
        "treatment_step": "Offer or review at most one concrete treatment action, only if downstream treatment readiness permits it.",
    }.get(move, "Gather one clinically useful missing detail with at most one focused question.")
    return (
        f"Response move: {move}. Response depth: {depth}. {guidance} "
        "Sound like a person in a real conversation; do not mechanically restate the phase plan."
    )


def decide_route(
    query: str,
    history: str,
    trace: TraceWriter,
    milestone_context: str = "",
) -> Dict[str, str]:
    route_prompt = TURN_MODE_PROMPT_MILESTONE.format(
        milestone_block=milestone_context if milestone_context else "(not available)",
        history_block=history if history else "(none)",
        user_msg=query,
    )
    raw = call_model(route_prompt, json_mode=True)
    mode = "analysis"
    reason = "default"
    response_move = ""
    depth = ""
    try:
        obj = json.loads(raw)
        mode = str(obj.get("mode", "analysis")).strip().lower()
        reason = str(obj.get("reason", "llm-router")).strip()
        response_move = str(obj.get("response_move", "")).strip().lower()
        depth = str(obj.get("depth", "")).strip().lower()
    except json.JSONDecodeError:
        lowered = query.lower().strip()
        greetings = ("hi", "hello", "hey", "good morning", "good afternoon", "how are you", "thanks")
        if any(lowered.startswith(item) for item in greetings) and len(lowered.split()) <= 8:
            mode = "chat"
            reason = "heuristic-greeting"
        else:
            mode = "analysis"
            reason = "heuristic-default"

    if mode in {"milestone", "clinical"}:
        mode = "analysis"
    if mode not in {"chat", "analysis"}:
        mode = "analysis"
    lowered = query.lower().strip()
    greeting = any(
        lowered.startswith(item)
        for item in ("hi", "hello", "hey", "good morning", "good afternoon", "how are you", "thanks")
    )
    if (
        mode == "chat"
        and not greeting
        and _LOW_INFORMATION_RE.match(query)
        and _CLINICAL_HISTORY_RE.search(history)
    ):
        mode = "analysis"
        response_move = "acknowledge"
        depth = "brief"
        reason = "short continuation inside clinical dialogue"
    if response_move not in VALID_RESPONSE_MOVES:
        response_move = _fallback_response_move(query, mode)
    if mode == "chat":
        response_move = "casual"
    elif _MEMORY_EXPLANATION_RE.search(query):
        response_move = "psychoeducation"
        if depth not in VALID_RESPONSE_DEPTHS:
            depth = "standard"
    if depth not in VALID_RESPONSE_DEPTHS:
        depth = "brief" if response_move in {"casual", "acknowledge", "reflect"} else "standard"
    trace(
        "route_decision",
        {
            "route_prompt": clip_text(route_prompt),
            "route_raw": clip_text(raw),
            "route_mode": mode,
            "response_move": response_move,
            "response_depth": depth,
            "route_reason": reason,
        },
    )
    return {
        "mode": mode,
        "response_move": response_move,
        "depth": depth,
        "reason": reason,
    }


def generate_chat_reply(query: str, history: str, route_reason: str, trace: TraceWriter) -> str:
    casual_prompt = build_casual_chat_prompt(query, history)
    gen = extract_final(call_model(casual_prompt, json_mode=False)).strip()
    used_fallback = False
    if looks_like_instruction_list(gen):
        fallback_prompt = (
            "Rewrite this into a natural doctor-to-patient reply.\n"
            "Constraints: 1-3 sentences; supportive but natural; no bullet points; no meta advice.\n\n"
            f"Patient message:\n{query}\n\n"
            f"Draft to rewrite:\n{gen}"
        )
        gen = extract_final(call_model(fallback_prompt, json_mode=False)).strip()
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
