from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from ..harness.adapters import CallableModelAdapter
from ..harness.registry import SkillRegistry
from ..harness.runner import HarnessRunner
from ..paths import (
    DEFAULT_ALERT_PATH,
    DEFAULT_KNOWLEDGE_TREE_PATH,
    DEFAULT_LOG_PATH,
    DEFAULT_MEMORY_PATH,
    DEFAULT_MILESTONE_PATH,
    DEFAULT_STATE_PATH,
    DEFAULT_TRACE_PATH,
    DEFAULT_TRANSCRIPT_PATH,
    repo_relative_path,
    resolve_repo_path,
)
from ..retrieval.knowledge_retriever import KnowledgeTreeRetriever
from ..retrieval.transcript_rag import load_segments, pick_refs
from ..safety import (
    ClinicalAlertNotifier,
    MoodAssessment,
    TreatmentReadiness,
    assess_patient_state,
    review_final_response,
)
from ..safety.risk import critical_risk_response, stopped_conversation_response
from ..services.helper_client import HelperApiClient
from ..skills.ocd_erp.skill import OcdErpSkill
from ..tracking.milestones import load_milestones, load_session_config
from . import session_logic as session_logic_module
from .session_logic import (
    build_turn_artifacts,
    clip_text,
    decide_route,
    default_progress_update,
    generate_analysis_reply,
    generate_chat_reply,
    serialize_knowledge_hits,
    summarize_dialogue_memory,
)
from .session_store import MemoryStore, Turn, log_debug


class DigitalDoctorSession:
    """Session resources and persistence wrapped by a versioned execution harness."""

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
        skill_id: str = "ocd_erp",
        skill_version: Optional[str] = None,
        skill_registry: Optional[SkillRegistry] = None,
        clinical_skill: Optional[object] = None,
        model_adapter: Optional[object] = None,
    ) -> None:
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

        self.model_adapter = model_adapter or CallableModelAdapter(
            model_call=lambda prompt, **kwargs: session_logic_module.call_model(prompt, **kwargs),
            adapter_id="prompt-model",
        )
        default_skill = OcdErpSkill(
            risk_assessor=lambda user_text, history="", trace=None: assess_patient_state(
                user_text,
                history,
                trace=trace,
            ),
            response_reviewer=lambda **kwargs: review_final_response(**kwargs),
            analysis_generator=lambda **kwargs: self._generate_analysis_reply(**kwargs),
            chat_generator=lambda query, history, reason: self._generate_chat_reply(
                query,
                history,
                reason,
            ),
        )
        self.skill_registry = skill_registry or SkillRegistry()
        if clinical_skill is not None:
            self.skill = clinical_skill
            if skill_registry is None:
                self.skill_registry.register(self.skill)
        else:
            if skill_registry is None:
                self.skill_registry.register(default_skill)
            self.skill = self.skill_registry.resolve(skill_id, skill_version)

        self.memory = MemoryStore(
            path=memory_path,
            window_size=8,
            default_user_id=self.user_id,
            summary_path=long_term_memory_path,
            summary_threshold_chars=memory_summary_threshold_chars,
        )
        self.notifier = ClinicalAlertNotifier(path=alert_path, webhook_url=alert_webhook_url)

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
        if (
            self.use_knowledge_tree
            and resolved_knowledge_tree_path
            and os.path.exists(resolved_knowledge_tree_path)
        ):
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
        self.tracker = self.skill.create_tracker(
            milestones,
            session_goal=goal,
            event_writer=self._trace,
        )
        self._restore_safety_state()
        self.harness_runner = HarnessRunner(self, self.skill, self.model_adapter)
        self._log_initialization(
            milestones=milestones,
            goal=goal,
            transcript_path=transcript_path,
            milestone_path=milestone_path,
            session_config_path=session_config_path,
            helper_api_url=helper_api_url,
            knowledge_tree_path=knowledge_tree_path,
        )

    def _log_initialization(
        self,
        *,
        milestones,
        goal: str,
        transcript_path: str,
        milestone_path: str,
        session_config_path: Optional[str],
        helper_api_url: str,
        knowledge_tree_path: Optional[str],
    ) -> None:
        self._log(f"Loaded {len(milestones)} milestones from {repo_relative_path(milestone_path)}")
        self._log(f"Built {len(self.tracker.phases)} therapy phases from skill policy")
        health = self.tracker.health()
        self._log(
            "planner initialized "
            f"health={health['status']} "
            f"target=P{health['current_phase_id']}: {health['current_phase']}",
            stage="milestone",
        )
        self._log(
            f"Loaded {len(self.segments)} reference segments from {repo_relative_path(transcript_path)}"
        )
        self._trace(
            "session_init",
            {
                "session_id": self.session_id,
                "harness": self.harness_runner.identity.to_dict(),
                "skill": self.skill.manifest.to_dict(),
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

    @staticmethod
    def _clip(text: str, limit: int = 4000) -> str:
        return clip_text(text, limit=limit)

    @staticmethod
    def _ensure_parent(path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _trace(self, event: str, payload: Dict[str, object]) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            **payload,
        }
        self._ensure_parent(self.trace_path)
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
        self._ensure_parent(self.state_path)
        with open(self.state_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _restore_safety_state(self) -> None:
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
        snapshot["harness"] = self.harness_runner.identity.to_dict()
        snapshot["skill"] = self.skill.manifest.to_dict()
        snapshot["safety_state"] = {
            "conversation_stopped": self.conversation_stopped,
            "mood": self.latest_mood.to_dict() if self.latest_mood else None,
            "treatment_readiness": self.latest_readiness.to_dict() if self.latest_readiness else None,
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

    @staticmethod
    def _serialize_knowledge_hits(knowledge_hits: Sequence[object]) -> List[Dict[str, object]]:
        return serialize_knowledge_hits(knowledge_hits)

    def _build_turn_artifacts(self, **kwargs) -> Dict[str, object]:
        return build_turn_artifacts(**kwargs)

    def _generate_analysis_reply(self, **kwargs) -> str:
        if "treatment_policy" in kwargs:
            kwargs["treatment_policy_block"] = kwargs.pop("treatment_policy")
        return generate_analysis_reply(
            **kwargs,
            trace=self._trace,
            model_adapter=self.model_adapter,
        )

    def _generate_chat_reply(self, query: str, history: str, route_reason: str) -> str:
        return generate_chat_reply(
            query,
            history,
            route_reason,
            trace=self._trace,
            model_adapter=self.model_adapter,
        )

    def _decide_route(self, query: str, history: str, milestone_context: str = "") -> Dict[str, str]:
        return decide_route(query, history, trace=self._trace, milestone_context=milestone_context)

    @staticmethod
    def stopped_response() -> str:
        return stopped_conversation_response()

    @staticmethod
    def critical_response() -> str:
        return critical_risk_response()

    def pick_references(self, query: str) -> List[str]:
        return pick_refs(query, self.segments) if self.segments else []

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
        return self.skill.phase_prerequisites_met(self.tracker)

    def _record_turn(self, query: str, reply: str, route: str) -> Dict[str, object]:
        if self.single_turn:
            return {"saved": False, "compacted": False}
        self.memory.append(Turn(self.user_id, self.episode_id, self.patient_role_label, query, kind=route))
        self.memory.append(Turn(self.user_id, self.episode_id, self.doctor_role_label, reply, kind=route))
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
        self,
        mood: MoodAssessment,
        query: str,
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
        return self.harness_runner.run_turn(query)


MilestoneSession = DigitalDoctorSession
