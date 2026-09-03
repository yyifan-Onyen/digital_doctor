"""OCD/ERP-specific dialogue routing and action planning."""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, Mapping

from .definition import ACTION_GUIDANCE, VALID_RESPONSE_DEPTHS, VALID_RESPONSE_MOVES
from .prompts import TURN_MODE_PROMPT_MILESTONE


TraceWriter = Callable[[str, Dict[str, object]], None]
ModelCall = Callable[..., str]

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


def _clip(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def fallback_response_move(query: str, mode: str) -> str:
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
    guidance = ACTION_GUIDANCE.get(move, ACTION_GUIDANCE["assess"])
    return (
        f"Response move: {move}. Response depth: {depth}. {guidance} "
        "Sound like a person in a real conversation; do not mechanically restate the phase plan."
    )


def decide_route(
    query: str,
    history: str,
    trace: TraceWriter,
    milestone_context: str,
    *,
    model_call: ModelCall,
) -> Dict[str, str]:
    route_prompt = TURN_MODE_PROMPT_MILESTONE.format(
        milestone_block=milestone_context if milestone_context else "(not available)",
        history_block=history if history else "(none)",
        user_msg=query,
    )
    raw = model_call(route_prompt, json_mode=True)
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
        greetings = (
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "how are you",
            "thanks",
        )
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
        for item in (
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "how are you",
            "thanks",
        )
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
        response_move = fallback_response_move(query, mode)
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
            "route_prompt": _clip(route_prompt),
            "route_raw": _clip(raw),
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


__all__ = [
    "VALID_RESPONSE_DEPTHS",
    "VALID_RESPONSE_MOVES",
    "decide_route",
    "fallback_response_move",
    "response_move_instructions",
]
