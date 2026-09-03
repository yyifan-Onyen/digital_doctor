from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..core.text_utils import clean_inline
from ..services.openai_client import call_model
from ..skills.ocd_erp.definition import (
    FORMULATION_FIELDS as DEFAULT_FORMULATION_FIELDS,
    PHASE_SPECS as DEFAULT_PHASE_SPECS,
)
from ..skills.ocd_erp.tracking_policy import (
    CONTEXT_POLICY_LINES as DEFAULT_CONTEXT_POLICY_LINES,
    apply_structured_phase_floors as default_phase_floor_policy,
    build_formulation_update_prompt as default_formulation_prompt_builder,
    build_phase_progress_prompt as default_phase_prompt_builder,
)


FORMULATION_FIELDS: List[Tuple[str, str]] = list(DEFAULT_FORMULATION_FIELDS)
TrackerEventWriter = Callable[[str, Dict[str, object]], None]
_VALID_PHASE_STATUSES = {"pending", "active", "completed", "blocked", "contraindicated"}
_BUY_IN_EVIDENCE_RE = re.compile(
    r"\b(i(?:'m| am) willing|i can try|that makes sense|it makes sense|i get why|"
    r"short[- ]term.{0,100}(?:long[- ]term|longer term)|"
    r"don'?t have to (?:avoid|wash|check)|do not have to (?:avoid|wash|check))\b",
    re.IGNORECASE,
)


@dataclass
class Milestone:
    milestone_id: int
    title: str
    description: str
    samples: List[str]


@dataclass
class Phase:
    phase_id: int
    slug: str
    title: str
    description: str
    goals: List[str] = field(default_factory=list)
    exit_criteria: List[str] = field(default_factory=list)
    legacy_milestone_ids: List[int] = field(default_factory=list)


@dataclass
class MilestoneState:
    status: str = "pending"
    first_turn: Optional[int] = None
    last_evidence: str = ""
    blocked_reason: str = ""
    contraindication_reason: str = ""


@dataclass
class FormulationFieldState:
    value: str = ""
    confidence: float = 0.0
    evidence: str = ""
    last_updated_turn: Optional[int] = None


@dataclass
class CaseFormulation:
    fields: Dict[str, FormulationFieldState] = field(default_factory=dict)
    field_definitions: Sequence[Tuple[str, str]] = field(
        default_factory=lambda: tuple(FORMULATION_FIELDS)
    )

    def __post_init__(self) -> None:
        if not self.fields:
            self.fields = {
                name: FormulationFieldState() for name, _ in self.field_definitions
            }

    def apply_updates(
        self,
        updates: Sequence[Dict[str, object]],
        turn_idx: int,
        min_confidence: float,
    ) -> List[Dict[str, object]]:
        applied: List[Dict[str, object]] = []
        labels = dict(self.field_definitions)
        for item in updates:
            field_name = str(item.get("field", "")).strip()
            if field_name not in self.fields:
                continue
            value = str(item.get("value", "")).strip()
            if not value:
                continue
            try:
                confidence = float(item.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            if confidence < min_confidence:
                continue

            evidence = str(item.get("evidence", "")).strip()
            current = self.fields[field_name]
            if current.value == value and confidence <= current.confidence:
                continue

            current.value = value
            current.confidence = confidence
            current.evidence = evidence
            current.last_updated_turn = turn_idx
            applied.append(
                {
                    "field": field_name,
                    "label": labels.get(field_name, field_name),
                    "value": value,
                    "confidence": round(confidence, 4),
                    "evidence": evidence,
                }
            )
        return applied

    def filled_fields(self) -> List[str]:
        return [name for name, state in self.fields.items() if state.value.strip()]

    def render_context(self) -> str:
        lines: List[str] = []
        for field_name, label in self.field_definitions:
            state = self.fields[field_name]
            if not state.value:
                continue
            lines.append(f"- {label}: {state.value}")
        return "\n".join(lines) if lines else "(not enough structured case data yet)"

    def snapshot(self) -> Dict[str, object]:
        payload = []
        for field_name, label in self.field_definitions:
            state = self.fields[field_name]
            payload.append(
                {
                    "field": field_name,
                    "label": label,
                    "value": state.value,
                    "confidence": round(state.confidence, 4),
                    "evidence": state.evidence,
                    "last_updated_turn": state.last_updated_turn,
                }
            )
        return {
            "filled_count": len(self.filled_fields()),
            "total_fields": len(self.field_definitions),
            "fields": payload,
        }


def load_milestones(path: str) -> List[Milestone]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Milestone file not found: {path}")

    header_re = re.compile(r"^###\s+\*\*(?:\([^)]*\)\s*)?Milestone\s+(\d+):\s*(.+?)\*\*")
    milestones: List[Milestone] = []
    cur_id: Optional[int] = None
    cur_title = ""
    cur_desc = ""
    cur_samples: List[str] = []
    in_samples = False

    with open(path, "r", encoding="utf-8-sig") as handle:
        lines = handle.readlines()

    def flush_current() -> None:
        nonlocal cur_id, cur_title, cur_desc, cur_samples, in_samples
        if cur_id is not None:
            milestones.append(
                Milestone(
                    milestone_id=cur_id,
                    title=cur_title.strip(),
                    description=cur_desc.strip(),
                    samples=[sample for sample in cur_samples if sample.strip()],
                )
            )
        cur_id = None
        cur_title = ""
        cur_desc = ""
        cur_samples = []
        in_samples = False

    for raw_line in lines:
        line = raw_line.strip()
        match = header_re.match(line)
        if match:
            flush_current()
            cur_id = int(match.group(1))
            cur_title = clean_inline(match.group(2))
            continue
        if cur_id is None:
            continue
        if line.startswith("**Description:**"):
            cur_desc = clean_inline(line.split("**Description:**", 1)[1])
            in_samples = False
            continue
        if line.startswith("**Samples:**"):
            in_samples = True
            continue
        if line.startswith("----------"):
            flush_current()
            continue
        if in_samples and line.startswith("-"):
            sample = clean_inline(line.lstrip("-").strip())
            if sample:
                cur_samples.append(sample)
            continue
        if (not in_samples) and line and not line.startswith("###"):
            if cur_desc:
                cur_desc += " " + clean_inline(line)

    flush_current()
    milestones.sort(key=lambda item: item.milestone_id)
    return milestones


def load_session_config(path: Optional[str]) -> Tuple[str, Optional[Set[int]]]:
    if not path:
        return ("Follow ERP phases consistently and progress through unresolved targets.", None)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Session config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        obj = json.load(handle)
    goal = str(obj.get("session_goal", "")).strip()
    if not goal:
        goal = "Follow ERP phases consistently and progress through unresolved targets."
    ids = obj.get("milestone_ids")
    if isinstance(ids, list):
        return goal, {int(item) for item in ids}
    return goal, None


def _phase_specs() -> List[Dict[str, object]]:
    # Kept as a compatibility function; the live source of truth is the
    # versioned OCD/ERP skill bundle's phase_graph.json.
    return [dict(item) for item in DEFAULT_PHASE_SPECS]


def build_phase_plan(
    milestones: Sequence[Milestone],
    phase_specs: Optional[Sequence[Dict[str, object]]] = None,
) -> List[Phase]:
    legacy_by_id = {item.milestone_id: item for item in milestones}
    phases: List[Phase] = []
    for spec in phase_specs or _phase_specs():
        goals = list(spec["goals"])
        legacy_ids: List[int] = []
        for legacy_id in spec["legacy_ids"]:
            legacy = legacy_by_id.get(int(legacy_id))
            if legacy is None:
                continue
            goals.append(f"M{legacy.milestone_id}: {legacy.description}")
            legacy_ids.append(legacy.milestone_id)
        phases.append(
            Phase(
                phase_id=int(spec["phase_id"]),
                slug=str(spec["slug"]),
                title=str(spec["title"]),
                description=str(spec["description"]),
                goals=goals,
                exit_criteria=list(spec["exit_criteria"]),
                legacy_milestone_ids=legacy_ids,
            )
        )
    return phases


class MilestoneTracker:
    def __init__(
        self,
        milestones: Sequence[Milestone],
        session_goal: str,
        event_writer: Optional[TrackerEventWriter] = None,
        *,
        phase_specs: Optional[Sequence[Dict[str, object]]] = None,
        formulation_fields: Optional[Sequence[Tuple[str, str]]] = None,
        formulation_prompt_builder=None,
        phase_prompt_builder=None,
        phase_floor_policy=None,
        context_policy_lines: Optional[Sequence[str]] = None,
    ):
        self.milestones: List[Milestone] = list(milestones)
        self.phase_specs = [dict(item) for item in (phase_specs or _phase_specs())]
        self.formulation_fields = list(formulation_fields or FORMULATION_FIELDS)
        self.phases: List[Phase] = build_phase_plan(milestones, self.phase_specs)
        self.session_goal = session_goal
        self.event_writer = event_writer
        self.turn_idx = 0
        self.state: Dict[int, MilestoneState] = {
            phase.phase_id: MilestoneState() for phase in self.phases
        }
        if self.phases:
            self.state[self.phases[0].phase_id].status = "active"
        self.formulation = CaseFormulation(field_definitions=tuple(self.formulation_fields))
        self.formulation_prompt_builder = (
            formulation_prompt_builder or default_formulation_prompt_builder
        )
        self.phase_prompt_builder = phase_prompt_builder or default_phase_prompt_builder
        self.phase_floor_policy = phase_floor_policy or default_phase_floor_policy
        self.context_policy_lines = tuple(
            context_policy_lines or DEFAULT_CONTEXT_POLICY_LINES
        )
        self.phase_completion_conf = float(os.getenv("PHASE_COMPLETION_CONF", "0.65"))
        self.phase_block_conf = float(os.getenv("PHASE_BLOCK_CONF", "0.70"))
        self.formulation_update_conf = float(os.getenv("FORMULATION_UPDATE_CONF", "0.60"))
        self._last_formulation_diagnostic: Dict[str, object] = {
            "status": "not_run",
            "parse_ok": None,
            "update_count": 0,
        }
        self._last_phase_diagnostic: Dict[str, object] = {
            "status": "not_run",
            "parse_ok": None,
            "phase_count": 0,
            "expected_phase_count": len(self.phases),
            "output_complete": None,
        }

    @staticmethod
    def _clip_for_log(text: str, limit: int = 6000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"... [truncated {len(text) - limit} chars]"

    def _emit(self, event: str, payload: Dict[str, object]) -> None:
        if self.event_writer is None:
            return
        self.event_writer(
            event,
            {
                "planner_turn_idx": self.turn_idx,
                **payload,
            },
        )

    def _formulation_summary(self) -> str:
        return self.formulation.render_context()

    def _phase_lines(self) -> str:
        lines: List[str] = []
        for phase in self.phases:
            current = self.state[phase.phase_id]
            goals = "; ".join(phase.goals[:3])
            lines.append(
                f"P{phase.phase_id} | slug: {phase.slug} | title: {phase.title} | "
                f"current_status: {current.status} | description: {phase.description} | goals: {goals}"
            )
        return "\n".join(lines)

    def _extract_formulation_updates(self, user_text: str, history: str) -> List[Dict[str, object]]:
        formulation_block = self._formulation_summary()
        prompt = self.formulation_prompt_builder(
            formulation_block,
            history,
            user_text,
            [name for name, _ in self.formulation_fields],
        )

        raw = call_model(prompt, json_mode=True)
        normalized: List[Dict[str, object]] = []
        parse_error = ""
        try:
            obj = json.loads(raw)
            items = obj.get("updates", [])
            if isinstance(items, list):
                seen = set()
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    field_name = str(item.get("field", "")).strip()
                    if field_name not in self.formulation.fields or field_name in seen:
                        continue
                    seen.add(field_name)
                    normalized.append(
                        {
                            "field": field_name,
                            "value": str(item.get("value", "")).strip(),
                            "confidence": item.get("confidence", 0.0),
                            "evidence": str(item.get("evidence", "")).strip(),
                        }
                    )
            else:
                parse_error = "updates must be a list"
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

        parse_ok = not parse_error
        self._last_formulation_diagnostic = {
            "status": "ok" if parse_ok else "degraded",
            "parse_ok": parse_ok,
            "update_count": len(normalized),
            "parse_error": parse_error,
        }
        self._emit(
            "milestone_formulation_inference",
            {
                "turn_idx_preview": self.turn_idx + 1,
                "prompt": self._clip_for_log(prompt),
                "raw": self._clip_for_log(raw),
                "parse_ok": parse_ok,
                "parse_error": parse_error,
                "updates": normalized,
            },
        )
        return normalized

    def observe_user_turn(self, user_text: str, history: str = "") -> Dict[str, object]:
        updates = self._extract_formulation_updates(user_text, history)
        applied = self.formulation.apply_updates(
            updates,
            turn_idx=self.turn_idx + 1,
            min_confidence=self.formulation_update_conf,
        )
        observation = {
            "formulation_updates": applied,
            "formulation_filled_count": len(self.formulation.filled_fields()),
            "formulation_total_fields": len(self.formulation_fields),
        }
        self._emit(
            "milestone_formulation_updated",
            {
                "turn_idx_preview": self.turn_idx + 1,
                **observation,
                "formulation": self.formulation.snapshot(),
            },
        )
        return observation

    def _infer_phase_progress(self, user_text: str, assistant_text: str) -> List[Dict[str, object]]:
        infer_prompt = self.phase_prompt_builder(
            self.session_goal,
            self._formulation_summary(),
            self._phase_lines(),
            user_text,
            assistant_text,
        )

        raw = call_model(infer_prompt, json_mode=True)
        normalized: List[Dict[str, object]] = []
        parse_error = ""
        try:
            obj = json.loads(raw)
            items = obj.get("phases", [])
            if isinstance(items, list):
                normalized = [item for item in items if isinstance(item, dict)]
            else:
                parse_error = "phases must be a list"
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

        returned_ids: Set[int] = set()
        for item in normalized:
            try:
                returned_ids.add(int(item.get("id")))
            except (TypeError, ValueError):
                continue
        expected_ids = {phase.phase_id for phase in self.phases}
        output_complete = returned_ids == expected_ids and len(normalized) == len(self.phases)
        parse_ok = not parse_error
        self._last_phase_diagnostic = {
            "status": "ok" if parse_ok and output_complete else "degraded",
            "parse_ok": parse_ok,
            "parse_error": parse_error,
            "phase_count": len(normalized),
            "expected_phase_count": len(self.phases),
            "returned_phase_ids": sorted(returned_ids),
            "output_complete": output_complete,
        }
        self._emit(
            "milestone_phase_inference",
            {
                "turn_idx": self.turn_idx,
                "prompt": self._clip_for_log(infer_prompt),
                "raw": self._clip_for_log(raw),
                **self._last_phase_diagnostic,
                "phases": normalized,
            },
        )
        return normalized

    def _normalize_phase_status(self, status: str, confidence: float) -> str:
        normalized = status.strip().lower()
        if normalized not in _VALID_PHASE_STATUSES:
            normalized = "active"
        if normalized == "completed" and confidence < self.phase_completion_conf:
            return "active"
        if normalized in {"blocked", "contraindicated"} and confidence < self.phase_block_conf:
            return "active"
        return normalized

    def _apply_structured_phase_floors(
        self,
        parsed: Dict[int, Dict[str, object]],
        user_text: str,
        assistant_text: str,
    ) -> List[Dict[str, object]]:
        filled = set(self.formulation.filled_fields())
        return self.phase_floor_policy(
            self.turn_idx,
            filled,
            parsed,
            user_text,
            _BUY_IN_EVIDENCE_RE,
        )

    def update(self, user_text: str, assistant_text: str) -> Dict[str, object]:
        target_before = self.next_target()
        statuses_before = {
            phase.phase_id: self.state[phase.phase_id].status for phase in self.phases
        }
        self.turn_idx += 1
        inferred = self._infer_phase_progress(user_text, assistant_text)
        parsed: Dict[int, Dict[str, object]] = {}
        for item in inferred:
            try:
                phase_id = int(item.get("id"))
            except Exception:
                continue
            try:
                confidence = float(item.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            status = self._normalize_phase_status(str(item.get("status", "active")), confidence)
            parsed[phase_id] = {
                "status": status,
                "confidence": confidence,
                "evidence": str(item.get("evidence", "")).strip(),
                "blocked_reason": str(item.get("blocked_reason", "")).strip(),
                "contraindication_reason": str(item.get("contraindication_reason", "")).strip(),
            }

        floor_overrides = self._apply_structured_phase_floors(parsed, user_text, assistant_text)
        if floor_overrides:
            self._emit(
                "milestone_phase_floor_applied",
                {
                    "overrides": floor_overrides,
                    "filled_formulation_fields": self.formulation.filled_fields(),
                },
            )

        top_scores = []
        for phase in self.phases:
            info = parsed.get(phase.phase_id, {"confidence": 0.0})
            top_scores.append(
                {
                    "id": phase.phase_id,
                    "title": phase.title,
                    "score": round(float(info.get("confidence", 0.0)), 4),
                }
            )
        top_scores.sort(key=lambda item: item["score"], reverse=True)
        top_scores = top_scores[:5]

        status_changes: List[Dict[str, str]] = []
        completed_now: List[int] = []
        blocked_now: List[int] = []
        contraindicated_now: List[int] = []
        first_unresolved_locked = False

        for phase in self.phases:
            current = self.state[phase.phase_id]
            status_before = current.status
            info = parsed.get(phase.phase_id, {})
            evidence = str(info.get("evidence", "")).strip()
            blocked_reason = str(info.get("blocked_reason", "")).strip()
            contraindication_reason = str(info.get("contraindication_reason", "")).strip()

            if status_before in {"completed", "contraindicated"}:
                new_status = status_before
            elif first_unresolved_locked:
                new_status = "pending"
            else:
                requested = str(info.get("status", status_before or "active")).strip().lower() or "active"
                if requested == "pending":
                    requested = "active"
                new_status = requested
                if new_status in {"active", "blocked"}:
                    first_unresolved_locked = True
                elif new_status == "contraindicated":
                    first_unresolved_locked = True

            if new_status != "blocked":
                blocked_reason = ""
            if new_status != "contraindicated":
                contraindication_reason = ""

            if new_status != "pending" and current.first_turn is None:
                current.first_turn = self.turn_idx
            if new_status != "pending":
                current.last_evidence = (evidence or assistant_text)[:280]
            current.status = new_status
            current.blocked_reason = blocked_reason
            current.contraindication_reason = contraindication_reason

            if status_before != current.status:
                status_changes.append(
                    {
                        "id": str(phase.phase_id),
                        "title": phase.title,
                        "from": status_before,
                        "to": current.status,
                    }
                )
                if current.status == "completed":
                    completed_now.append(phase.phase_id)
                elif current.status == "blocked":
                    blocked_now.append(phase.phase_id)
                elif current.status == "contraindicated":
                    contraindicated_now.append(phase.phase_id)

        if not any(
            self.state[phase.phase_id].status in {"active", "blocked"} for phase in self.phases
        ):
            next_target = self.next_target()
            if next_target is not None and self.state[next_target.phase_id].status == "pending":
                self.state[next_target.phase_id].status = "active"
                status_changes.append(
                    {
                        "id": str(next_target.phase_id),
                        "title": next_target.title,
                        "from": "pending",
                        "to": "active",
                    }
                )

        completed_ids = [
            phase.phase_id for phase in self.phases if self.state[phase.phase_id].status == "completed"
        ]
        next_target = self.next_target()
        blocked_phase = next(
            (phase for phase in self.phases if self.state[phase.phase_id].status == "blocked"),
            None,
        )

        target_after = self.next_target()
        health = self.health()
        transition = {
            "advanced": (
                target_before is not None
                and (target_after is None or target_before.phase_id != target_after.phase_id)
            ),
            "from_phase_id": target_before.phase_id if target_before else None,
            "from_phase": target_before.title if target_before else "Completed",
            "to_phase_id": target_after.phase_id if target_after else None,
            "to_phase": target_after.title if target_after else "Completed",
        }
        update = {
            "turn_idx": self.turn_idx,
            "covered_now": completed_now,
            "partial_now": blocked_now,
            "blocked_now": blocked_now,
            "contraindicated_now": contraindicated_now,
            "covered_count": len(completed_ids),
            "total": len(self.phases),
            "next_target": f"P{next_target.phase_id}: {next_target.title}" if next_target else "All phases completed",
            "current_phase": (
                f"P{next_target.phase_id}: {next_target.title}" if next_target else "Completed"
            ),
            "blocked_phase": (
                f"P{blocked_phase.phase_id}: {blocked_phase.title}" if blocked_phase else ""
            ),
            "top_scores": top_scores,
            "status_changes": status_changes,
            "transition": transition,
            "milestone_health": health,
            "formulation_filled_count": len(self.formulation.filled_fields()),
            "formulation_total_fields": len(self.formulation_fields),
        }
        self._emit(
            "milestone_state_transition",
            {
                "statuses_before": statuses_before,
                "statuses_after": {
                    phase.phase_id: self.state[phase.phase_id].status for phase in self.phases
                },
                "status_changes": status_changes,
                "transition": transition,
                "next_target": update["next_target"],
            },
        )
        self._emit("milestone_health", health)
        return update

    def next_target(self) -> Optional[Phase]:
        for phase in self.phases:
            status = self.state[phase.phase_id].status
            if status not in {"completed", "contraindicated"}:
                return phase
        return None

    def following_phase(self, phase: Optional[Phase] = None) -> Optional[Phase]:
        target = phase or self.next_target()
        if target is None:
            return None
        for index, candidate in enumerate(self.phases):
            if candidate.phase_id == target.phase_id:
                return self.phases[index + 1] if index + 1 < len(self.phases) else None
        return None

    def health(self) -> Dict[str, object]:
        """Return an auditable structural and model-output health check."""
        violations: List[str] = []
        phase_ids = [phase.phase_id for phase in self.phases]
        if len(phase_ids) != len(set(phase_ids)):
            violations.append("duplicate phase ids")

        unresolved_seen = False
        focus_ids: List[int] = []
        for phase in self.phases:
            status = self.state[phase.phase_id].status
            if status not in _VALID_PHASE_STATUSES:
                violations.append(f"P{phase.phase_id} has invalid status {status!r}")
            resolved = status in {"completed", "contraindicated"}
            if unresolved_seen and resolved:
                violations.append(f"P{phase.phase_id} is resolved after an unresolved earlier phase")
            if not resolved:
                unresolved_seen = True
            if status in {"active", "blocked"}:
                focus_ids.append(phase.phase_id)

        target = self.next_target()
        if len(focus_ids) > 1:
            violations.append(f"multiple active or blocked phases: {focus_ids}")
        if target is not None and focus_ids != [target.phase_id]:
            violations.append(
                f"focus {focus_ids or 'none'} does not match earliest unresolved P{target.phase_id}"
            )
        if target is None and focus_ids:
            violations.append("completed plan still has an active or blocked phase")

        phase_model_ok = self._last_phase_diagnostic.get("status") in {"not_run", "ok"}
        structural_ok = not violations
        status = "not_run" if self.turn_idx == 0 else (
            "healthy" if structural_ok and phase_model_ok else "degraded"
        )
        return {
            "status": status,
            "healthy": structural_ok and phase_model_ok,
            "structural_ok": structural_ok,
            "violations": violations,
            "current_phase_id": target.phase_id if target else None,
            "current_phase": target.title if target else "Completed",
            "focus_phase_ids": focus_ids,
            "formulation_inference": dict(self._last_formulation_diagnostic),
            "phase_inference": dict(self._last_phase_diagnostic),
        }

    def render_context(self) -> str:
        completed = [phase for phase in self.phases if self.state[phase.phase_id].status == "completed"]
        blocked = [phase for phase in self.phases if self.state[phase.phase_id].status == "blocked"]
        contraindicated = [
            phase for phase in self.phases if self.state[phase.phase_id].status == "contraindicated"
        ]
        next_target = self.next_target()
        following = self.following_phase(next_target)
        completed_txt = ", ".join([f"P{phase.phase_id}" for phase in completed]) if completed else "none yet"
        blocked_txt = ", ".join([f"P{phase.phase_id}" for phase in blocked]) if blocked else "none"
        contraindicated_txt = (
            ", ".join([f"P{phase.phase_id}" for phase in contraindicated]) if contraindicated else "none"
        )
        next_txt = (
            f"P{next_target.phase_id}: {next_target.title}. {next_target.description}"
            if next_target
            else "All phases completed. Consolidate gains and self-directed ERP."
        )
        blocked_reason = ""
        if next_target is not None:
            blocked_reason = self.state[next_target.phase_id].blocked_reason
        target_status = self.state[next_target.phase_id].status if next_target else "completed"
        lines = [
            f"Session goal: {self.session_goal}",
            f"Phase coverage: {len(completed)}/{len(self.phases)}",
            f"Completed phases: {completed_txt}",
            f"Blocked phases: {blocked_txt}",
            f"Contraindicated phases: {contraindicated_txt}",
            f"Current priority phase: {next_txt}",
            f"Current phase status: {target_status}",
        ]
        if next_target is not None:
            lines.append("Current phase goals: " + "; ".join(next_target.goals))
            lines.append("Current phase exit criteria: " + "; ".join(next_target.exit_criteria))
            current_evidence = self.state[next_target.phase_id].last_evidence.strip()
            lines.append(f"Current phase evidence: {current_evidence or '(none recorded yet)'}")
            if following is not None:
                lines.append(
                    f"Next phase after completion: P{following.phase_id}: {following.title}. "
                    f"{following.description}"
                )
                lines.append("Next phase opening goal: " + following.goals[0])
            else:
                lines.append("Next phase after completion: maintenance and consolidation")
        if blocked_reason:
            lines.append(f"Current block: {blocked_reason}")
        lines.append("Structured case formulation:")
        lines.append(self._formulation_summary())
        lines.extend(self.context_policy_lines)
        return "\n".join(lines)

    def snapshot(self) -> Dict[str, object]:
        phases = []
        for phase in self.phases:
            state = self.state[phase.phase_id]
            phases.append(
                {
                    "id": phase.phase_id,
                    "slug": phase.slug,
                    "title": phase.title,
                    "status": state.status,
                    "first_turn": state.first_turn,
                    "last_evidence": state.last_evidence,
                    "blocked_reason": state.blocked_reason,
                    "contraindication_reason": state.contraindication_reason,
                    "legacy_milestone_ids": phase.legacy_milestone_ids,
                }
            )
        next_target = self.next_target()
        return {
            "turn_idx": self.turn_idx,
            "session_goal": self.session_goal,
            "coverage": f"{sum(1 for item in phases if item['status'] == 'completed')}/{len(phases)}",
            "next_target": next_target.phase_id if next_target else None,
            "phases": phases,
            "formulation": self.formulation.snapshot(),
            "health": self.health(),
        }


PhasePlanner = MilestoneTracker
PhaseState = MilestoneState


__all__ = [
    "CaseFormulation",
    "FORMULATION_FIELDS",
    "Milestone",
    "MilestoneState",
    "MilestoneTracker",
    "Phase",
    "PhasePlanner",
    "PhaseState",
    "TrackerEventWriter",
    "build_phase_plan",
    "load_milestones",
    "load_session_config",
]
