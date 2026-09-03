from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from ..paths import (
    DEFAULT_KNOWLEDGE_TREE_PATH,
    DEFAULT_ALERT_PATH,
    DEFAULT_LOG_PATH,
    DEFAULT_MEMORY_PATH,
    DEFAULT_MILESTONE_PATH,
    repo_relative_path,
    resolve_repo_path,
    DEFAULT_STATE_PATH,
    DEFAULT_TRACE_PATH,
    DEFAULT_TRANSCRIPT_PATH,
)
from ..prompt import (
    build_helper_prompt,
    build_helper_query_prompt,
)
from ..services.helper_client import HelperApiClient
from ..services.openai_client import call_model
from ..tracking.milestones import MilestoneTracker, load_milestones, load_session_config
from ..safety import (
    ClinicalAlertNotifier,
    MoodAssessment,
    TreatmentReadiness,
    assess_patient_state,
    assess_treatment_readiness,
    review_final_response,
)
from ..safety.risk import critical_risk_response, stopped_conversation_response
from ..safety.treatment import (
    enforce_high_risk_treatment_limit,
    enforce_treatment_buffer,
    treatment_policy_block,
)
from ..retrieval.knowledge_retriever import KnowledgeTreeRetriever
from ..retrieval.transcript_rag import load_segments, pick_refs
from .session_logic import (
    build_turn_artifacts,
    clip_text,
    decide_route,
    default_progress_update,
    generate_analysis_reply,
    generate_chat_reply,
    response_move_instructions,
    serialize_knowledge_hits,
    summarize_dialogue_memory,
)
from .session_store import MemoryStore, Turn, log_debug
from .text_utils import extract_final


class DigitalDoctorSession:
    def __init__(
        self,
        transcript_path: str = DEFAULT_TRANSCRIPT_PATH,
        milestone_path: str = DEFAULT_MILESTONE_PATH,
        helper_api_url: str = "http://localhost:8001/helper/generate",
        helper_api_key: Optional[str] = None,
        session_config_path: Optional[str] = None,
        user_id: str = "user",
        episode_id: str = "session",
        patient_role_label: str = "user",
        doctor_role_label: str = "assistant",
        memory_path: str = DEFAULT_MEMORY_PATH,
        long_term_memory_path: Optional[str] = None,
        state_path: str = DEFAULT_STATE_PATH,
        log_path: str = DEFAULT_LOG_PATH,
        trace_path: str = DEFAULT_TRACE_PATH,
        alert_path: str = DEFAULT_ALERT_PATH,
        alert_webhook_url: Optional[str] = None,
        memory_summary_threshold_chars: Optional[int] = None,
        treatment_min_context_turns: Optional[int] = None,
        single_turn: bool = False,
        use_helper_model: bool = True,
        knowledge_tree_path: Optional[str] = DEFAULT_KNOWLEDGE_TREE_PATH,
        use_knowledge_tree: bool = True,
    ):
        self.user_id = user_id
        self.episode_id = episode_id
        self.patient_role_label = patient_role_label
        self.doctor_role_label = doctor_role_label
        self.single_turn = single_turn
        self.use_helper_model = use_helper_model
        self.use_knowledge_tree = use_knowledge_tree
        self.treatment_min_context_turns = treatment_min_context_turns
        self.conversation_stopped = False
        self.latest_alert: Optional[Dict[str, object]] = None
        self.latest_mood: Optional[MoodAssessment] = None
        self.latest_readiness: Optional[TreatmentReadiness] = None
        self.use_safety_check = os.getenv("USE_SAFETY_CHECK", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.state_path = str(resolve_repo_path(state_path))
        self.log_path = str(resolve_repo_path(log_path))
        self.trace_path = str(resolve_repo_path(trace_path))
        self.session_id = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.memory = MemoryStore(
            path=memory_path,
            window_size=8,
            default_user_id=self.user_id,
            summary_path=long_term_memory_path,
            summary_threshold_chars=memory_summary_threshold_chars,
        )
        self.notifier = ClinicalAlertNotifier(
            path=alert_path,
            webhook_url=alert_webhook_url,
        )
        resolved_transcript_path = str(resolve_repo_path(transcript_path))
        resolved_milestone_path = str(resolve_repo_path(milestone_path))
        resolved_session_config_path = (
            str(resolve_repo_path(session_config_path)) if session_config_path else None
        )
        self.segments = load_segments(resolved_transcript_path)
        self.helper_client = (
            HelperApiClient(helper_api_url, api_key=helper_api_key) if self.use_helper_model else None
        )
        self.knowledge_retriever: Optional[KnowledgeTreeRetriever] = None
        self.knowledge_tree_path = knowledge_tree_path
        resolved_knowledge_tree_path = (
            str(resolve_repo_path(knowledge_tree_path)) if knowledge_tree_path else None
        )
        if self.use_knowledge_tree and resolved_knowledge_tree_path and os.path.exists(resolved_knowledge_tree_path):
            self.knowledge_retriever = KnowledgeTreeRetriever(resolved_knowledge_tree_path)
        elif self.use_knowledge_tree:
            self._log(
                f"Knowledge tree enabled but file not found: {repo_relative_path(knowledge_tree_path)}"
            )

        goal, active_ids = load_session_config(resolved_session_config_path)
        milestones = load_milestones(resolved_milestone_path)
        if active_ids is not None:
            milestones = [item for item in milestones if item.milestone_id in active_ids]
        if not milestones:
            raise RuntimeError("No milestones loaded. Check milestone file or session config milestone_ids.")
        self.tracker = MilestoneTracker(
            milestones,
            session_goal=goal,
            event_writer=self._trace,
        )
        self._restore_safety_state()
        self._log(
            f"Loaded {len(milestones)} milestones from {repo_relative_path(milestone_path)}"
        )
        self._log(f"Built {len(self.tracker.phases)} therapy phases from the legacy milestone set")
        initial_health = self.tracker.health()
        self._log(
            "planner initialized "
            f"health={initial_health['status']} "
            f"target=P{initial_health['current_phase_id']}: {initial_health['current_phase']}",
            stage="milestone",
        )
        self._log(
            f"Loaded {len(self.segments)} reference segments from {repo_relative_path(transcript_path)}"
        )
        self._trace(
            "session_init",
            {
                "session_id": self.session_id,
                "episode_id": self.episode_id,
                "patient_role_label": self.patient_role_label,
                "doctor_role_label": self.doctor_role_label,
                "single_turn": self.single_turn,
                "memory_summary_threshold_chars": self.memory.summary_threshold_chars,
                "treatment_min_context_turns": self.treatment_min_context_turns
                or int(os.getenv("TREATMENT_MIN_CONTEXT_TURNS", "3")),
                "helper_api_url": helper_api_url,
                "helper_enabled": self.use_helper_model,
                "knowledge_tree_enabled": self.knowledge_retriever is not None,
                "knowledge_tree_path": repo_relative_path(knowledge_tree_path),
                "transcript_path": repo_relative_path(transcript_path),
                "milestone_path": repo_relative_path(milestone_path),
                "session_config_path": repo_relative_path(session_config_path),
                "session_goal": goal,
                "milestone_count": len(milestones),
                "phase_count": len(self.tracker.phases),
                "phases": [
                    {
                        "id": item.phase_id,
                        "slug": item.slug,
                        "title": item.title,
                        "description": item.description,
                        "goal_count": len(item.goals),
                        "legacy_milestone_ids": item.legacy_milestone_ids,
                    }
                    for item in self.tracker.phases
                ],
                "milestones": [
                    {
                        "id": item.milestone_id,
                        "title": item.title,
                        "description": item.description,
                        "sample_count": len(item.samples),
                        "sample_preview": item.samples[:2],
                    }
                    for item in milestones
                ],
                "segment_count": len(self.segments),
            },
        )

    def _log(self, message: str, stage: Optional[str] = None) -> None:
        log_debug(message, path=self.log_path, stage=stage)

    def _clip(self, text: str, limit: int = 4000) -> str:
        return clip_text(text, limit=limit)

    def _trace(self, event: str, payload: Dict[str, object]) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            **payload,
        }
        os.makedirs(os.path.dirname(self.trace_path), exist_ok=True)
        with open(self.trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _append_state(self, query: str, reply: str, update: Dict[str, object]) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "episode_id": self.episode_id,
            "query": query,
            "reply": reply,
            "update": update,
            "snapshot": self.tracker.snapshot(),
        }
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _restore_safety_state(self) -> None:
        """Keep a critical stop active when an episode is reloaded without a reset."""
        if not os.path.exists(self.state_path):
            return
        latest: Optional[Dict[str, object]] = None
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if str(record.get("episode_id", "")) == self.episode_id:
                        latest = record
        except (OSError, json.JSONDecodeError):
            return
        update = latest.get("update", {}) if isinstance(latest, dict) else {}
        if not isinstance(update, dict):
            return
        self.conversation_stopped = bool(update.get("conversation_stopped", False))
        alert = update.get("clinical_alert")
        if isinstance(alert, dict):
            self.latest_alert = alert

    def snapshot(self) -> Dict[str, object]:
        snapshot = self.tracker.snapshot()
        memory = self._memory_context("")
        alert_summary: Optional[Dict[str, object]] = None
        if self.latest_alert:
            alert_summary = {
                "alert_id": self.latest_alert.get("alert_id", ""),
                "status": self.latest_alert.get("status", ""),
                "timestamp": self.latest_alert.get("timestamp", ""),
            }
        snapshot["safety_state"] = {
            "conversation_stopped": self.conversation_stopped,
            "mood": self.latest_mood.to_dict() if self.latest_mood else None,
            "treatment_readiness": (
                self.latest_readiness.to_dict() if self.latest_readiness else None
            ),
            "latest_alert": alert_summary,
        }
        snapshot["memory"] = {
            "archived_turn_count": memory["archived_turn_count"],
            "compaction_count": memory["compaction_count"],
            "topics": memory["topics"],
            "has_long_term_summary": bool(memory["long_term_summary"]),
        }
        return snapshot

    def _default_progress_update(self, route: str) -> Dict[str, object]:
        return default_progress_update(self.tracker, route)

    def _serialize_knowledge_hits(self, knowledge_hits: Sequence[object]) -> List[Dict[str, object]]:
        return serialize_knowledge_hits(knowledge_hits)

    def _build_turn_artifacts(
        self,
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
        return build_turn_artifacts(
            route=route,
            history=history,
            milestone_context=milestone_context,
            milestone_snapshot_before=milestone_snapshot_before,
            refs=refs,
            knowledge_hits=knowledge_hits,
            helper_query=helper_query,
            helper_answer=helper_answer,
            source_candidates=source_candidates,
            milestone_snapshot_after=milestone_snapshot_after,
            pre_turn_observation=pre_turn_observation,
        )

    def _generate_analysis_reply(
        self,
        query: str,
        history: str,
        history_block: str,
        milestone_context: str,
        milestone_block: str,
        refs_block: str,
        helper_block: str,
        knowledge_block: str,
        treatment_policy: str,
        response_move_block: str,
        candidate_name: str,
    ) -> str:
        return generate_analysis_reply(
            query=query,
            history=history,
            history_block=history_block,
            milestone_context=milestone_context,
            milestone_block=milestone_block,
            refs_block=refs_block,
            helper_block=helper_block,
            knowledge_block=knowledge_block,
            treatment_policy_block=treatment_policy,
            response_move_block=response_move_block,
            candidate_name=candidate_name,
            trace=self._trace,
        )

    def _decide_route(
        self,
        query: str,
        history: str,
        milestone_context: str = "",
    ) -> Dict[str, str]:
        return decide_route(
            query,
            history,
            trace=self._trace,
            milestone_context=milestone_context,
        )

    def _apply_safety(
        self,
        query: str,
        draft_reply: str,
        history: str,
        readiness: TreatmentReadiness,
        mood: MoodAssessment,
        response_move: str,
    ) -> Tuple[str, Optional[Dict[str, object]], Dict[str, object]]:
        """Apply the treatment buffer, clinical review, and a final fail-closed check."""
        treatment_step_selected = response_move == "treatment_step"
        action_allowed = bool(readiness.allowed and treatment_step_selected)
        limited_reply, high_risk_before = enforce_high_risk_treatment_limit(draft_reply)
        buffered_reply, before_meta = enforce_treatment_buffer(
            limited_reply,
            readiness,
            treatment_step_selected=treatment_step_selected,
        )
        if not self.use_safety_check:
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
        verdict = review_final_response(
            user_text=query,
            draft_reply=buffered_reply,
            history=history,
            formulation_context=self.tracker.formulation.render_context(),
            treatment_allowed=action_allowed,
            mood_assessment=mood.to_dict(),
            trace=self._trace,
        )
        if verdict.action != "allow":
            self._log(
                f"Safety gate: action={verdict.action} risk={verdict.risk_level} "
                f"categories={verdict.categories}"
            )
        limited_final, high_risk_after = enforce_high_risk_treatment_limit(verdict.final_reply)
        final_reply, after_meta = enforce_treatment_buffer(
            limited_final,
            readiness,
            force_risk_hold=False,
            treatment_step_selected=treatment_step_selected,
        )
        safety_meta = verdict.to_dict()
        high_risk_changed = bool(
            high_risk_before["changed"] or high_risk_after["changed"]
        )
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

    def _memory_context(self, query: str) -> Dict[str, object]:
        if self.single_turn:
            return {
                "rendered": "",
                "long_term_summary": "",
                "topics": [],
                "reminders": [],
                "recalled": [],
                "reminder_needed": False,
                "recent_turn_count": 0,
                "archived_turn_count": 0,
                "compaction_count": 0,
            }
        return self.memory.context(self.user_id, self.episode_id, query=query)

    def _treatment_phase_prerequisites_met(self) -> bool:
        required_slugs = {"assessment", "formulation", "buy_in"}
        required_phases = [
            phase for phase in self.tracker.phases if phase.slug in required_slugs
        ]
        return bool(required_phases) and all(
            self.tracker.state[phase.phase_id].status == "completed"
            for phase in required_phases
        )

    def _record_turn(self, query: str, reply: str, route: str) -> Dict[str, object]:
        if self.single_turn:
            return {"saved": False, "compacted": False}
        self.memory.append(
            Turn(self.user_id, self.episode_id, self.patient_role_label, query, kind=route)
        )
        self.memory.append(
            Turn(self.user_id, self.episode_id, self.doctor_role_label, reply, kind=route)
        )
        formulation_topics = [
            f"{name}: {state.value}"
            for name, state in self.tracker.formulation.fields.items()
            if state.value.strip()
        ]
        reminders: List[str] = []
        if route == "safety_stop":
            formulation_topics.append("critical safety escalation")
            reminders.append("Treatment dialogue is paused pending urgent human review.")
        topic_update = self.memory.update_topics(
            self.user_id,
            self.episode_id,
            formulation_topics,
            reminders,
        )
        compaction = self.memory.compact_if_needed(
            self.user_id,
            self.episode_id,
            summarize_dialogue_memory,
        )
        context = self.memory.context(self.user_id, self.episode_id)
        metadata = {
            "saved": True,
            **compaction,
            "topic_ledger_changed": topic_update["changed"],
            "topics": context["topics"],
            "reminders": context["reminders"],
            "archived_turn_count": context["archived_turn_count"],
            "long_term_summary": context["long_term_summary"],
        }
        self._trace("memory_updated", metadata)
        return metadata

    def _notify_if_needed(
        self, mood: MoodAssessment, query: str
    ) -> Optional[Dict[str, object]]:
        if not mood.notify_human:
            return None
        alert = self.notifier.notify(
            assessment=mood,
            user_text=query,
            user_id=self.user_id,
            episode_id=self.episode_id,
            session_id=self.session_id,
        )
        self.latest_alert = alert
        self._trace(
            "clinical_alert_created",
            {
                "alert_id": alert.get("alert_id", ""),
                "status": alert.get("status", ""),
                "risk_level": mood.risk_level,
            },
        )
        return alert

    def handle_query(self, query: str) -> Tuple[str, Dict[str, object]]:
        self._trace("turn_start", {"turn_query": query})
        memory_context = self._memory_context(query)
        history = str(memory_context["rendered"])

        if self.conversation_stopped:
            reply = stopped_conversation_response()
            update = self._default_progress_update(route="safety_stop")
            update.update(
                {
                    "conversation_stopped": True,
                    "clinical_alert": self.latest_alert,
                    "memory_recall": {
                        "recalled": memory_context["recalled"],
                        "reminder_needed": memory_context["reminder_needed"],
                    },
                    "safety": {
                        "action": "crisis",
                        "risk_level": "critical",
                        "categories": ["existing_safety_stop"],
                        "escalated": True,
                        "changed": True,
                    },
                }
            )
            update["memory"] = self._record_turn(query, reply, route="safety_stop")
            self._append_state(query, reply, update)
            self._trace("turn_end", {"route": "safety_stop", "conversation_stopped": True})
            return reply, update

        mood = assess_patient_state(query, history, trace=self._trace)
        self.latest_mood = mood
        clinical_alert = self._notify_if_needed(mood, query)
        if mood.stop_conversation:
            self.conversation_stopped = True
            reply = critical_risk_response()
            clinical_turns = (
                self.tracker.turn_idx + 1
                if self.single_turn
                else self.memory.clinical_user_turn_count(
                    self.user_id, self.episode_id, self.patient_role_label
                )
                + 1
            )
            readiness = assess_treatment_readiness(
                clinical_turns=clinical_turns,
                formulation_fields=self.tracker.formulation.fields,
                mood=mood,
                minimum_turns=self.treatment_min_context_turns,
                phase_prerequisites_met=self._treatment_phase_prerequisites_met(),
            )
            self.latest_readiness = readiness
            update = self._default_progress_update(route="safety_stop")
            update.update(
                {
                    "mood": mood.to_dict(),
                    "risk_state": mood.to_dict(),
                    "treatment_readiness": readiness.to_dict(),
                    "conversation_stopped": True,
                    "clinical_alert": clinical_alert,
                    "memory_recall": {
                        "recalled": memory_context["recalled"],
                        "reminder_needed": memory_context["reminder_needed"],
                    },
                    "safety": {
                        "action": "crisis",
                        "risk_level": "critical",
                        "categories": list(mood.categories),
                        "confidence": mood.confidence,
                        "rationale": mood.rationale,
                        "escalated": True,
                        "changed": True,
                    },
                }
            )
            update["artifacts"] = {
                "route": "safety_stop",
                "history_excerpt": self._clip(history, limit=1500),
                "mood_risk_assessment": mood.to_dict(),
                "treatment_readiness": readiness.to_dict(),
            }
            update["memory"] = self._record_turn(query, reply, route="safety_stop")
            self._append_state(query, reply, update)
            self._trace(
                "turn_end",
                {
                    "route": "safety_stop",
                    "conversation_stopped": True,
                    "alert_id": (clinical_alert or {}).get("alert_id", ""),
                },
            )
            return reply, update

        routing_milestone_context = self.tracker.render_context()
        route = self._decide_route(query, history, routing_milestone_context)
        if route["mode"] == "analysis":
            pre_turn_observation = self.tracker.observe_user_turn(query, history)
        else:
            pre_turn_observation = {
                "formulation_updates": [],
                "formulation_filled_count": len(self.tracker.formulation.filled_fields()),
                "formulation_total_fields": len(self.tracker.formulation.fields),
                "skipped_for_chat": True,
            }
        history_block = f"Dialogue memory:\n{history}\n" if history else ""
        milestone_context = self.tracker.render_context()
        milestone_snapshot_before = self.tracker.snapshot()
        milestone_block = f"Phase planner context:\n{milestone_context}\n"
        response_move_block = response_move_instructions(route)
        if route["mode"] == "analysis":
            clinical_turns = (
                self.tracker.turn_idx + 1
                if self.single_turn
                else self.memory.clinical_user_turn_count(
                    self.user_id, self.episode_id, self.patient_role_label
                )
                + 1
            )
        else:
            clinical_turns = self.tracker.turn_idx
        readiness = assess_treatment_readiness(
            clinical_turns=clinical_turns,
            formulation_fields=self.tracker.formulation.fields,
            mood=mood,
            minimum_turns=self.treatment_min_context_turns,
            phase_prerequisites_met=self._treatment_phase_prerequisites_met(),
        )
        self.latest_readiness = readiness
        policy = treatment_policy_block(readiness, route["response_move"])
        self._trace(
            "turn_context_built",
            {
                "turn_idx_preview": self.tracker.turn_idx + 1,
                "history_chars": len(history),
                "history": self._clip(history, limit=2000),
                "milestone_context": milestone_context,
                "pre_turn_observation": pre_turn_observation,
                "route": route["mode"],
                "route_decision": dict(route),
                "memory_recall": {
                    "recalled": memory_context["recalled"],
                    "reminder_needed": memory_context["reminder_needed"],
                },
                "mood_risk_assessment": mood.to_dict(),
                "treatment_readiness": readiness.to_dict(),
                "helper_enabled": self.use_helper_model,
                "knowledge_tree_enabled": self.knowledge_retriever is not None,
            },
        )

        if route["mode"] == "chat":
            draft = generate_chat_reply(query, history, route["reason"], trace=self._trace)
            gen, safety_meta, treatment_output = self._apply_safety(
                query, draft, history, readiness, mood, route["response_move"]
            )
            update = self._default_progress_update(route="chat")
            update["route_decision"] = dict(route)
            update["mood"] = mood.to_dict()
            update["risk_state"] = mood.to_dict()
            update["treatment_readiness"] = readiness.to_dict()
            update["treatment_output_gate"] = treatment_output
            update["conversation_stopped"] = False
            update["clinical_alert"] = clinical_alert
            update["memory_recall"] = {
                "recalled": memory_context["recalled"],
                "reminder_needed": memory_context["reminder_needed"],
            }
            if safety_meta is not None:
                update["safety"] = safety_meta
            update["artifacts"] = self._build_turn_artifacts(
                route="chat",
                history=history,
                milestone_context=milestone_context,
                milestone_snapshot_before=milestone_snapshot_before,
                refs=[],
                knowledge_hits=[],
                milestone_snapshot_after=self.tracker.snapshot(),
                pre_turn_observation=pre_turn_observation,
            )
            if safety_meta is not None and safety_meta.get("changed"):
                update["artifacts"]["safety"] = safety_meta
                update["artifacts"]["pre_safety_reply"] = draft
            update["artifacts"]["mood_risk_assessment"] = mood.to_dict()
            update["artifacts"]["treatment_readiness"] = readiness.to_dict()
            update["artifacts"]["treatment_output_gate"] = treatment_output
            update["artifacts"]["route_decision"] = dict(route)
            update.update(pre_turn_observation)
            update["memory"] = self._record_turn(query, gen, route="chat")
            self._append_state(query, gen, update)
            self._log(
                f"turn={update['turn_idx']} route=chat health={update['milestone_health']['status']} "
                f"target={update['current_phase']} progression=skipped",
                stage="milestone",
            )
            self._trace(
                "turn_end",
                {
                    "turn_idx": update["turn_idx"],
                    "coverage": f"{update['covered_count']}/{update['total']}",
                    "route": "chat",
                },
            )
            return gen, update

        helper_answer = ""
        helper_query = ""
        if self.use_helper_model and self.helper_client is not None:
            helper_query_prompt = build_helper_query_prompt(
                query,
                history,
                milestone_context,
                response_move=route["response_move"],
            )
            helper_query_raw = call_model(helper_query_prompt, json_mode=True)
            try:
                helper_obj = json.loads(helper_query_raw)
                helper_query = str(helper_obj.get("helper_prompt", helper_query_raw))
            except json.JSONDecodeError:
                helper_query = helper_query_raw
            helper_query = extract_final(helper_query)
            self._trace(
                "helper_query_generated",
                {
                    "helper_query_prompt": self._clip(helper_query_prompt),
                    "helper_query_raw": self._clip(helper_query_raw),
                    "helper_query_text": self._clip(helper_query),
                },
            )

            helper_prompt = build_helper_prompt(
                helper_query,
                history,
                milestone_context,
                response_move=route["response_move"],
            )
            helper_raw = self.helper_client.generate(helper_prompt)
            helper_answer = extract_final(helper_raw)
            self._trace(
                "helper_response_received",
                {
                    "helper_prompt": self._clip(helper_prompt),
                    "helper_raw": self._clip(helper_raw),
                    "helper_answer": self._clip(helper_answer),
                },
            )
        else:
            self._trace("helper_skipped", {"reason": "helper_disabled"})

        refs = pick_refs(query, self.segments) if self.segments else []
        refs_block = "Reference cases:\n" + "\n".join(refs) + "\n\n" if refs else ""
        helper_block = f"Helper model suggestion:\n{helper_answer}\n\n" if helper_answer else ""
        knowledge_hits = []
        knowledge_block = ""
        if self.knowledge_retriever is not None:
            knowledge_hits = self.knowledge_retriever.retrieve(
                query=query,
                history=history,
                milestone_context=milestone_context,
            )
            formatted = self.knowledge_retriever.format_block(knowledge_hits)
            knowledge_block = f"{formatted}\n\n" if formatted else ""
        self._trace(
            "retrieval_complete",
            {
                "ref_count": len(refs),
                "refs": refs,
                "knowledge_hit_count": len(knowledge_hits),
                "knowledge_hits": self._serialize_knowledge_hits(knowledge_hits),
            },
        )

        transcript_candidate = self._generate_analysis_reply(
            query=query,
            history=history,
            history_block=history_block,
            milestone_context=milestone_context,
            milestone_block=milestone_block,
            refs_block=refs_block,
            helper_block=helper_block,
            knowledge_block="",
            treatment_policy=policy,
            response_move_block=response_move_block,
            candidate_name="transcript",
        )
        knowledge_candidate = ""
        if self.knowledge_retriever is not None:
            knowledge_candidate = self._generate_analysis_reply(
                query=query,
                history=history,
                history_block=history_block,
                milestone_context=milestone_context,
                milestone_block=milestone_block,
                refs_block="",
                helper_block=helper_block,
                knowledge_block=knowledge_block,
                treatment_policy=policy,
                response_move_block=response_move_block,
                candidate_name="knowledge_tree",
            )
        if self.knowledge_retriever is None:
            draft_combined = transcript_candidate
        else:
            draft_combined = self._generate_analysis_reply(
                query=query,
                history=history,
                history_block=history_block,
                milestone_context=milestone_context,
                milestone_block=milestone_block,
                refs_block=refs_block,
                helper_block=helper_block,
                knowledge_block=knowledge_block,
                treatment_policy=policy,
                response_move_block=response_move_block,
                candidate_name="combined",
            )
        gen, safety_meta, treatment_output = self._apply_safety(
            query, draft_combined, history, readiness, mood, route["response_move"]
        )

        update = self.tracker.update(query, gen)
        update["route"] = "analysis"
        update["route_decision"] = dict(route)
        update.update(pre_turn_observation)
        update["mood"] = mood.to_dict()
        update["risk_state"] = mood.to_dict()
        update["treatment_readiness"] = readiness.to_dict()
        update["treatment_output_gate"] = treatment_output
        update["conversation_stopped"] = False
        update["clinical_alert"] = clinical_alert
        update["memory_recall"] = {
            "recalled": memory_context["recalled"],
            "reminder_needed": memory_context["reminder_needed"],
        }
        if safety_meta is not None:
            update["safety"] = safety_meta
        source_candidates = {
            "transcript": transcript_candidate,
            "knowledge_tree": knowledge_candidate,
            "combined": gen,
            "combined_pre_safety": draft_combined,
        }
        update["source_candidates"] = source_candidates
        update["artifacts"] = self._build_turn_artifacts(
            route="analysis",
            history=history,
            milestone_context=milestone_context,
            milestone_snapshot_before=milestone_snapshot_before,
            refs=refs,
            knowledge_hits=knowledge_hits,
            helper_query=helper_query,
            helper_answer=helper_answer,
            source_candidates=source_candidates,
            milestone_snapshot_after=self.tracker.snapshot(),
            pre_turn_observation=pre_turn_observation,
        )
        if safety_meta is not None and safety_meta.get("changed"):
            update["artifacts"]["safety"] = safety_meta
            update["artifacts"]["pre_safety_reply"] = draft_combined
        update["artifacts"]["mood_risk_assessment"] = mood.to_dict()
        update["artifacts"]["treatment_readiness"] = readiness.to_dict()
        update["artifacts"]["treatment_output_gate"] = treatment_output
        update["artifacts"]["route_decision"] = dict(route)
        update["memory"] = self._record_turn(query, gen, route="analysis")
        self._append_state(query, gen, update)
        self._log(f"Turn {update['turn_idx']}: coverage {update['covered_count']}/{update['total']}")
        transition = update["transition"]
        phase_diagnostic = update["milestone_health"]["phase_inference"]
        self._log(
            f"turn={update['turn_idx']} route=analysis health={update['milestone_health']['status']} "
            f"phase_output_complete={phase_diagnostic['output_complete']} "
            f"target={update['current_phase']} advanced={transition['advanced']} "
            f"status_changes={len(update['status_changes'])}",
            stage="milestone",
        )
        self._trace("tracker_updated", {"update": update, "snapshot": self.tracker.snapshot()})

        self._trace(
            "turn_end",
            {
                "turn_idx": update["turn_idx"],
                "coverage": f"{update['covered_count']}/{update['total']}",
                "route": "analysis",
            },
        )
        return gen, update


MilestoneSession = DigitalDoctorSession
