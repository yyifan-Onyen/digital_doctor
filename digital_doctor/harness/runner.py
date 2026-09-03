"""Turn lifecycle executor shared by all clinical skills."""

from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

from ..core.text_utils import extract_final
from .contracts import (
    ActionPlan,
    EvidenceBundle,
    GenerationSpec,
    HARNESS_VERSION,
    StateDelta,
    TurnContext,
)
from .platform_safety import PlatformSafetyGate


class HarnessRunner:
    """Execute a turn while keeping clinical policy behind a typed skill API."""

    def __init__(self, session, skill, model_adapter) -> None:
        self.session = session
        self.skill = skill
        self.model_adapter = model_adapter
        self.identity = skill.identity(HARNESS_VERSION, model_adapter.adapter_id)
        self.platform_safety = PlatformSafetyGate(
            stopped_response=session.stopped_response,
            critical_response=session.critical_response,
        )

    def _context(self, query: str, memory_context: Dict[str, object]) -> TurnContext:
        session = self.session
        return TurnContext(
            session_id=session.session_id,
            episode_id=session.episode_id,
            user_id=session.user_id,
            query=query,
            history=str(memory_context["rendered"]),
            turn_index=session.tracker.turn_idx + 1,
            memory_recall={
                "recalled": memory_context["recalled"],
                "reminder_needed": memory_context["reminder_needed"],
            },
        )

    def _clinical_turns(self, mode: str) -> int:
        session = self.session
        if mode != "analysis":
            return session.tracker.turn_idx
        if session.single_turn:
            return session.tracker.turn_idx + 1
        return session.memory.clinical_user_turn_count(
            session.user_id,
            session.episode_id,
            session.patient_role_label,
        ) + 1

    def _decorate_update(
        self,
        update: Dict[str, object],
        *,
        plan: Optional[ActionPlan] = None,
        state_delta: Optional[StateDelta] = None,
        evidence: Optional[EvidenceBundle] = None,
    ) -> None:
        update["harness"] = self.identity.to_dict()
        update["skill"] = self.skill.manifest.to_dict()
        if plan is not None:
            update["action_plan"] = plan.to_dict()
        if state_delta is not None:
            update["state_delta"] = state_delta.to_dict()
        if evidence is not None:
            update["evidence_bundle"] = evidence.to_dict()

    def _emit_distillation_record(
        self,
        context: TurnContext,
        update: Dict[str, object],
        reply: str,
    ) -> None:
        session = self.session
        record = {
            "record_version": 1,
            "identity": self.identity.to_dict(),
            "student_input": {
                "history": context.history,
                "patient_message": context.query,
            },
            "privileged_skill_context": {
                "clinical_state_before": (
                    update.get("artifacts", {}).get("milestone_snapshot_before_turn", {})
                    if isinstance(update.get("artifacts"), dict)
                    else {}
                ),
                "state_delta": update.get("state_delta", {}),
                "action_plan": update.get("action_plan", {}),
                "treatment_readiness": update.get("treatment_readiness", {}),
                "evidence": update.get("evidence_bundle", {}),
            },
            "teacher_target": {
                "response": reply,
                "clinical_state_after": session.tracker.snapshot(),
                "safety": update.get("safety", {}),
            },
        }
        session._trace("distillation_record", record)

    def _commit(
        self,
        context: TurnContext,
        reply: str,
        update: Dict[str, object],
        route: str,
    ) -> Tuple[str, Dict[str, object]]:
        session = self.session
        update["memory"] = session._record_turn(context.query, reply, route=route)
        session._append_state(context.query, reply, update)
        self._emit_distillation_record(context, update, reply)
        session._trace(
            "harness_turn_committed",
            {
                "identity": self.identity.to_dict(),
                "route": route,
                "action_plan": update.get("action_plan", {}),
                "conversation_stopped": update.get("conversation_stopped", False),
            },
        )
        return reply, update

    def _existing_stop(
        self,
        context: TurnContext,
        memory_context: Dict[str, object],
    ) -> Tuple[str, Dict[str, object]]:
        session = self.session
        decision = self.platform_safety.preflight(True, None)
        reply = decision.response
        update = session._default_progress_update(route="safety_stop")
        update.update(
            {
                "conversation_stopped": True,
                "clinical_alert": session.latest_alert,
                "memory_recall": dict(context.memory_recall),
                "safety": {
                    "action": "crisis",
                    "risk_level": "critical",
                    "categories": [decision.reason],
                    "escalated": True,
                    "changed": True,
                },
            }
        )
        self._decorate_update(update)
        session._trace("turn_end", {"route": "safety_stop", "conversation_stopped": True})
        return self._commit(context, reply, update, "safety_stop")

    def _critical_stop(
        self,
        context: TurnContext,
        memory_context: Dict[str, object],
        mood,
        clinical_alert: Optional[Dict[str, object]],
    ) -> Tuple[str, Dict[str, object]]:
        session = self.session
        session.conversation_stopped = True
        decision = self.platform_safety.preflight(False, mood)
        readiness = self.skill.assess_readiness(
            self._clinical_turns("analysis"),
            session.tracker,
            mood,
            session.treatment_min_context_turns,
        )
        session.latest_readiness = readiness
        plan = ActionPlan(
            mode="safety_stop",
            action="stop",
            depth="brief",
            reason=decision.reason,
            allowed_actions=["stop"],
        )
        update = session._default_progress_update(route="safety_stop")
        update.update(
            {
                "mood": mood.to_dict(),
                "risk_state": mood.to_dict(),
                "treatment_readiness": readiness.to_dict(),
                "conversation_stopped": True,
                "clinical_alert": clinical_alert,
                "memory_recall": dict(context.memory_recall),
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
            "history_excerpt": session._clip(context.history, limit=1500),
            "mood_risk_assessment": mood.to_dict(),
            "treatment_readiness": readiness.to_dict(),
        }
        self._decorate_update(update, plan=plan)
        session._trace(
            "turn_end",
            {
                "route": "safety_stop",
                "conversation_stopped": True,
                "alert_id": (clinical_alert or {}).get("alert_id", ""),
            },
        )
        return self._commit(context, decision.response, update, "safety_stop")

    def _retrieve_evidence(
        self,
        context: TurnContext,
        milestone_context: str,
        plan: ActionPlan,
    ) -> Tuple[EvidenceBundle, str, str]:
        session = self.session
        helper_query = ""
        helper_answer = ""
        if session.use_helper_model and session.helper_client is not None:
            helper_query_prompt = self.skill.helper_query_prompt(
                context.query,
                context.history,
                milestone_context,
                plan,
            )
            helper_query_raw = self.model_adapter.generate(
                GenerationSpec(
                    stage="helper_query",
                    prompt=helper_query_prompt,
                    json_mode=True,
                    candidate_name="helper_query",
                )
            )
            try:
                helper_obj = json.loads(helper_query_raw)
                helper_query = str(helper_obj.get("helper_prompt", helper_query_raw))
            except json.JSONDecodeError:
                helper_query = helper_query_raw
            helper_query = extract_final(helper_query)
            session._trace(
                "helper_query_generated",
                {
                    "helper_query_prompt": session._clip(helper_query_prompt),
                    "helper_query_raw": session._clip(helper_query_raw),
                    "helper_query_text": session._clip(helper_query),
                },
            )
            helper_prompt = self.skill.helper_prompt(
                helper_query,
                context.history,
                milestone_context,
                plan,
            )
            helper_raw = session.helper_client.generate(helper_prompt)
            helper_answer = extract_final(helper_raw)
            session._trace(
                "helper_response_received",
                {
                    "helper_prompt": session._clip(helper_prompt),
                    "helper_raw": session._clip(helper_raw),
                    "helper_answer": session._clip(helper_answer),
                },
            )
        else:
            session._trace("helper_skipped", {"reason": "helper_disabled"})

        refs = session.pick_references(context.query)
        knowledge_hits = []
        knowledge_block = ""
        if session.knowledge_retriever is not None:
            knowledge_hits = session.knowledge_retriever.retrieve(
                query=context.query,
                history=context.history,
                milestone_context=milestone_context,
            )
            formatted = session.knowledge_retriever.format_block(knowledge_hits)
            knowledge_block = f"{formatted}\n\n" if formatted else ""
        serialized_hits = session._serialize_knowledge_hits(knowledge_hits)
        evidence = EvidenceBundle(
            transcript_refs=list(refs),
            knowledge_hits=serialized_hits,
            helper_query=helper_query,
            helper_answer=helper_answer,
        )
        session._trace(
            "skill_evidence_collected",
            {"identity": self.identity.to_dict(), **evidence.to_dict()},
        )
        session._trace(
            "retrieval_complete",
            {
                "ref_count": len(refs),
                "refs": refs,
                "knowledge_hit_count": len(knowledge_hits),
                "knowledge_hits": serialized_hits,
            },
        )
        return evidence, knowledge_block, helper_answer

    def _review_and_finalize(
        self,
        context: TurnContext,
        draft: str,
        readiness: object,
        mood: object,
        plan: ActionPlan,
    ) -> Tuple[str, Optional[Dict[str, object]], Dict[str, object]]:
        """Run the skill review, then enforce the harness-owned final gate."""
        session = self.session
        try:
            final, safety_meta, treatment_output = self.skill.review_and_gate(
                context.query,
                draft,
                context.history,
                session.tracker,
                readiness,
                mood,
                plan,
                session.use_safety_check,
                session._trace,
            )
            treatment_output = dict(treatment_output or {})
        except Exception as exc:
            session._trace(
                "skill_review_failed",
                {
                    "identity": self.identity.to_dict(),
                    "error_type": type(exc).__name__,
                },
            )
            final = (
                "I can't safely provide a clinical response from this system right now. "
                "Please pause and contact a licensed clinician for support."
            )
            safety_meta = {
                "action": "escalate",
                "risk_level": "elevated",
                "categories": ["skill_review_unavailable"],
                "rationale": "The clinical skill review failed, so the harness failed closed.",
                "changed": True,
                "escalated": True,
            }
            treatment_output = {
                "skill_review_failed": True,
                "final_check": {"advice_detected": True},
            }

        final_gate = self.platform_safety.finalize(
            final,
            mood,
            plan,
            treatment_output,
        )
        treatment_output["platform_final_gate"] = {
            "changed": final_gate.changed,
            "reason": final_gate.reason,
        }
        if final_gate.changed:
            safety_meta = dict(safety_meta or {})
            categories = [str(item) for item in safety_meta.get("categories", [])]
            if final_gate.reason not in categories:
                categories.append(final_gate.reason)
            is_critical = "critical" in final_gate.reason
            safety_meta.update(
                {
                    "action": "escalate" if is_critical else "revise",
                    "categories": categories,
                    "rationale": "The harness-owned final safety gate replaced the skill output.",
                    "changed": True,
                    "escalated": is_critical,
                }
            )
        return final_gate.reply, safety_meta, treatment_output

    def run_turn(self, query: str) -> Tuple[str, Dict[str, object]]:
        session = self.session
        session._trace("turn_start", {"turn_query": query})
        session._trace(
            "harness_turn_start",
            {"turn_query": query, "identity": self.identity.to_dict()},
        )
        memory_context = session._memory_context(query)
        context = self._context(query, memory_context)

        if session.conversation_stopped:
            return self._existing_stop(context, memory_context)

        mood = self.skill.assess_risk(context, session._trace)
        session.latest_mood = mood
        clinical_alert = session._notify_if_needed(mood, query)
        if self.platform_safety.preflight(False, mood).stop:
            return self._critical_stop(context, memory_context, mood, clinical_alert)

        plan = self.skill.plan(context, session.tracker, session._decide_route)
        state_delta = self.skill.observe(
            context,
            session.tracker,
            clinical=plan.mode == "analysis",
        )
        history_block = f"Dialogue memory:\n{context.history}\n" if context.history else ""
        milestone_context = session.tracker.render_context()
        milestone_snapshot_before = session.tracker.snapshot()
        milestone_block = f"Phase planner context:\n{milestone_context}\n"
        response_move_block = self.skill.response_instructions(plan)
        readiness = self.skill.assess_readiness(
            self._clinical_turns(plan.mode),
            session.tracker,
            mood,
            session.treatment_min_context_turns,
        )
        session.latest_readiness = readiness
        plan = self.platform_safety.authorize(plan, readiness)
        policy = self.skill.treatment_policy(readiness, plan)
        session._trace(
            "skill_action_planned",
            {
                "identity": self.identity.to_dict(),
                "action_plan": plan.to_dict(),
                "state_delta": state_delta.to_dict(),
            },
        )
        session._trace(
            "turn_context_built",
            {
                "turn_idx_preview": session.tracker.turn_idx + 1,
                "history_chars": len(context.history),
                "history": session._clip(context.history, limit=2000),
                "milestone_context": milestone_context,
                "pre_turn_observation": state_delta.metadata,
                "route": plan.mode,
                "route_decision": plan.to_route_dict(),
                "action_plan": plan.to_dict(),
                "memory_recall": dict(context.memory_recall),
                "mood_risk_assessment": mood.to_dict(),
                "treatment_readiness": readiness.to_dict(),
                "helper_enabled": session.use_helper_model,
                "knowledge_tree_enabled": session.knowledge_retriever is not None,
                "identity": self.identity.to_dict(),
            },
        )

        if plan.mode == "chat":
            draft = self.skill.generate_chat(
                context.query,
                context.history,
                plan.reason,
            )
            final, safety_meta, treatment_output = self._review_and_finalize(
                context,
                draft,
                readiness,
                mood,
                plan,
            )
            update = session._default_progress_update(route="chat")
            update.update(state_delta.metadata)
            update.update(
                {
                    "route_decision": plan.to_route_dict(),
                    "mood": mood.to_dict(),
                    "risk_state": mood.to_dict(),
                    "treatment_readiness": readiness.to_dict(),
                    "treatment_output_gate": treatment_output,
                    "conversation_stopped": False,
                    "clinical_alert": clinical_alert,
                    "memory_recall": dict(context.memory_recall),
                }
            )
            if safety_meta is not None:
                update["safety"] = safety_meta
            update["artifacts"] = session._build_turn_artifacts(
                route="chat",
                history=context.history,
                milestone_context=milestone_context,
                milestone_snapshot_before=milestone_snapshot_before,
                refs=[],
                knowledge_hits=[],
                milestone_snapshot_after=session.tracker.snapshot(),
                pre_turn_observation=state_delta.metadata,
            )
            if safety_meta is not None and safety_meta.get("changed"):
                update["artifacts"]["safety"] = safety_meta
                update["artifacts"]["pre_safety_reply"] = draft
            update["artifacts"].update(
                {
                    "mood_risk_assessment": mood.to_dict(),
                    "treatment_readiness": readiness.to_dict(),
                    "treatment_output_gate": treatment_output,
                    "route_decision": plan.to_route_dict(),
                }
            )
            self._decorate_update(update, plan=plan, state_delta=state_delta)
            session._log(
                f"turn={update['turn_idx']} route=chat health={update['milestone_health']['status']} "
                "target=" + str(update["current_phase"]) + " progression=skipped",
                stage="milestone",
            )
            session._trace(
                "turn_end",
                {
                    "turn_idx": update["turn_idx"],
                    "coverage": f"{update['covered_count']}/{update['total']}",
                    "route": "chat",
                },
            )
            return self._commit(context, final, update, "chat")

        evidence, knowledge_block, helper_answer = self._retrieve_evidence(
            context,
            milestone_context,
            plan,
        )
        refs_block = (
            "Reference cases:\n" + "\n".join(evidence.transcript_refs) + "\n\n"
            if evidence.transcript_refs
            else ""
        )
        helper_block = f"Helper model suggestion:\n{helper_answer}\n\n" if helper_answer else ""

        transcript_candidate = self.skill.generate_analysis(
            query=context.query,
            history=context.history,
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
        if session.knowledge_retriever is not None:
            knowledge_candidate = self.skill.generate_analysis(
                query=context.query,
                history=context.history,
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
        if session.knowledge_retriever is None:
            draft_combined = transcript_candidate
        else:
            draft_combined = self.skill.generate_analysis(
                query=context.query,
                history=context.history,
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
        final, safety_meta, treatment_output = self._review_and_finalize(
            context,
            draft_combined,
            readiness,
            mood,
            plan,
        )

        update = session.tracker.update(context.query, final)
        update["route"] = "analysis"
        update["route_decision"] = plan.to_route_dict()
        update.update(state_delta.metadata)
        update.update(
            {
                "mood": mood.to_dict(),
                "risk_state": mood.to_dict(),
                "treatment_readiness": readiness.to_dict(),
                "treatment_output_gate": treatment_output,
                "conversation_stopped": False,
                "clinical_alert": clinical_alert,
                "memory_recall": dict(context.memory_recall),
            }
        )
        if safety_meta is not None:
            update["safety"] = safety_meta
        source_candidates = {
            "transcript": transcript_candidate,
            "knowledge_tree": knowledge_candidate,
            "combined": final,
            "combined_pre_safety": draft_combined,
        }
        update["source_candidates"] = source_candidates
        update["artifacts"] = session._build_turn_artifacts(
            route="analysis",
            history=context.history,
            milestone_context=milestone_context,
            milestone_snapshot_before=milestone_snapshot_before,
            refs=evidence.transcript_refs,
            knowledge_hits=[],
            helper_query=evidence.helper_query,
            helper_answer=evidence.helper_answer,
            source_candidates=source_candidates,
            milestone_snapshot_after=session.tracker.snapshot(),
            pre_turn_observation=state_delta.metadata,
        )
        update["artifacts"]["knowledge_hits"] = evidence.knowledge_hits
        if safety_meta is not None and safety_meta.get("changed"):
            update["artifacts"]["safety"] = safety_meta
            update["artifacts"]["pre_safety_reply"] = draft_combined
        update["artifacts"].update(
            {
                "mood_risk_assessment": mood.to_dict(),
                "treatment_readiness": readiness.to_dict(),
                "treatment_output_gate": treatment_output,
                "route_decision": plan.to_route_dict(),
            }
        )
        self._decorate_update(
            update,
            plan=plan,
            state_delta=state_delta,
            evidence=evidence,
        )
        session._log(f"Turn {update['turn_idx']}: coverage {update['covered_count']}/{update['total']}")
        transition = update["transition"]
        phase_diagnostic = update["milestone_health"]["phase_inference"]
        session._log(
            f"turn={update['turn_idx']} route=analysis health={update['milestone_health']['status']} "
            f"phase_output_complete={phase_diagnostic['output_complete']} "
            f"target={update['current_phase']} advanced={transition['advanced']} "
            f"status_changes={len(update['status_changes'])}",
            stage="milestone",
        )
        session._trace("tracker_updated", {"update": update, "snapshot": session.tracker.snapshot()})
        session._trace(
            "turn_end",
            {
                "turn_idx": update["turn_idx"],
                "coverage": f"{update['covered_count']}/{update['total']}",
                "route": "analysis",
            },
        )
        return self._commit(context, final, update, "analysis")


__all__ = ["HarnessRunner"]
