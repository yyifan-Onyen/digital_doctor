from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from digital_doctor.core.session import DigitalDoctorSession
from digital_doctor.core.session_store import MemoryStore, Turn
from digital_doctor.safety.notifications import ClinicalAlertNotifier
from digital_doctor.safety.review import review_final_response
from digital_doctor.safety.risk import MoodAssessment, assess_patient_state
from digital_doctor.safety.treatment import (
    assess_treatment_readiness,
    enforce_high_risk_treatment_limit,
    enforce_treatment_buffer,
)
from digital_doctor.tracking.milestones import Milestone


STABLE = MoodAssessment(
    mood="calm",
    stability="stable",
    risk_level="low",
    confidence=0.9,
    treatment_allowed=True,
)


class CumulativeMemoryTests(unittest.TestCase):
    def test_compaction_persists_summary_topics_and_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            memory_path = root / "turns.jsonl"
            summary_path = root / "long_term.json"
            store = MemoryStore(
                path=str(memory_path),
                summary_path=str(summary_path),
                window_size=2,
                summary_threshold_chars=20,
            )
            turns = [
                Turn("u", "e", "patient", "Door handles trigger contamination fears."),
                Turn("u", "e", "doctor", "We mapped the trigger and the washing ritual."),
                Turn("u", "e", "patient", "I wash my hands three times."),
                Turn("u", "e", "doctor", "We are still gathering context."),
            ]
            for turn in turns:
                store.append(turn)

            result = store.compact_if_needed(
                "u",
                "e",
                lambda prior, chunk: {
                    "summary": "Door handles trigger contamination fears followed by repeated washing.",
                    "topics": ["door-handle contamination", "repeated washing"],
                    "reminders": ["The trigger and washing ritual were mapped."],
                },
            )

            self.assertTrue(result["compacted"])
            self.assertTrue(summary_path.exists())
            reloaded = MemoryStore(
                path=str(memory_path),
                summary_path=str(summary_path),
                window_size=2,
                summary_threshold_chars=20,
            )
            context = reloaded.context("u", "e", "What did we discuss about washing?")
            self.assertIn("Door handles", context["long_term_summary"])
            self.assertIn("repeated washing", context["topics"])
            self.assertTrue(context["reminder_needed"])
            self.assertTrue(context["recalled"])

    def test_forgetting_detection_allows_natural_adverbs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = MemoryStore(
                path=str(root / "turns.jsonl"),
                summary_path=str(root / "summary.json"),
                window_size=2,
            )
            store.append(Turn("u", "e", "patient", "The bathroom doorknob triggered me."))
            store.append(Turn("u", "e", "doctor", "We identified an urge to wash."))
            store.append(Turn("u", "e", "patient", "I avoided the bathroom afterward."))
            store.append(Turn("u", "e", "doctor", "We kept mapping the pattern."))

            context = store.context(
                "u",
                "e",
                "I honestly can't remember exactly why compulsions are unhelpful.",
            )

            self.assertTrue(context["reminder_needed"])
            self.assertTrue(context["recalled"])


class TreatmentBufferTests(unittest.TestCase):
    def test_requires_multiple_turns_context_and_stable_mood(self) -> None:
        fields = {"obsession": "contamination", "trigger": "door", "compulsion": "washing"}
        early = assess_treatment_readiness(1, fields, STABLE, minimum_turns=3)
        ready = assess_treatment_readiness(3, fields, STABLE, minimum_turns=3)
        unstable = MoodAssessment(
            mood="overwhelmed",
            stability="unstable",
            risk_level="high",
            treatment_allowed=False,
            notify_human=True,
        )
        held = assess_treatment_readiness(4, fields, unstable, minimum_turns=3)

        self.assertFalse(early.allowed)
        self.assertTrue(ready.allowed)
        self.assertFalse(held.allowed)
        self.assertEqual(held.stage, "paused_for_safety")
        held_reply, held_meta = enforce_treatment_buffer(
            "Take several slow breaths before we continue.", held
        )
        self.assertTrue(held_meta["forced_risk_hold"])
        self.assertIn("pausing that part", held_reply)

    def test_phase_prerequisites_keep_treatment_locked(self) -> None:
        readiness = assess_treatment_readiness(
            5,
            {"obsession": "contamination", "trigger": "door", "compulsion": "washing"},
            STABLE,
            minimum_turns=3,
            phase_prerequisites_met=False,
        )

        self.assertFalse(readiness.allowed)
        self.assertEqual(readiness.stage, "collecting_context")
        self.assertIn("phase_prerequisites_incomplete", readiness.reasons)

    def test_output_gate_replaces_premature_erp_advice(self) -> None:
        readiness = assess_treatment_readiness(
            1,
            {"obsession": "contamination", "trigger": "door", "compulsion": "washing"},
            STABLE,
            minimum_turns=3,
        )
        reply, metadata = enforce_treatment_buffer(
            "Would you be willing to try a short delay before washing next time?",
            readiness,
        )
        self.assertTrue(metadata["changed"])
        self.assertNotIn("delay before washing", reply)
        self.assertIn("Before we decide on any treatment step", reply)

    def test_readiness_does_not_authorize_exposure_without_treatment_move(self) -> None:
        readiness = assess_treatment_readiness(
            5,
            {"obsession": "contamination", "trigger": "door", "compulsion": "washing"},
            STABLE,
            minimum_turns=3,
            phase_prerequisites_met=True,
        )
        reply, metadata = enforce_treatment_buffer(
            "Try touching the doorknob and then do not wash your hands.",
            readiness,
            treatment_step_selected=False,
        )

        self.assertTrue(readiness.allowed)
        self.assertTrue(metadata["changed"])
        self.assertFalse(metadata["advice_allowed"])
        self.assertNotIn("doorknob", reply)
        self.assertIn("rather than jumping ahead", reply)

    def test_high_risk_treatment_is_blocked_even_when_context_is_ready(self) -> None:
        reply, metadata = enforce_high_risk_treatment_limit(
            "You should stop your prescribed medication before the exposure."
        )
        self.assertTrue(metadata["changed"])
        self.assertIn("can't safely guide", reply)


class MoodAndEscalationTests(unittest.TestCase):
    def test_heuristic_separates_imminent_intent_from_intrusive_thought(self) -> None:
        with patch("digital_doctor.safety.risk.call_model", side_effect=RuntimeError("offline")):
            critical = assess_patient_state("I have a plan to kill myself tonight.")
            obsession = assess_patient_state(
                "I have an unwanted intrusive thought that I might hurt myself, but I don't want to."
            )
        self.assertTrue(critical.stop_conversation)
        self.assertEqual(critical.risk_level, "critical")
        self.assertFalse(obsession.stop_conversation)

    def test_ordinary_ocd_anxiety_is_not_instability_when_model_is_unavailable(self) -> None:
        with patch("digital_doctor.safety.risk.call_model", side_effect=RuntimeError("offline")):
            anxious = assess_patient_state(
                "I'm really anxious and disgusted after touching the doorknob, but I can keep talking."
            )
            unstable = assess_patient_state("I can't think straight or cope right now.")

        self.assertEqual(anxious.stability, "stable")
        self.assertEqual(anxious.risk_level, "low")
        self.assertTrue(anxious.treatment_allowed)
        self.assertEqual(unstable.stability, "unstable")
        self.assertFalse(unstable.treatment_allowed)

    def test_model_strained_state_can_continue_when_not_high_risk(self) -> None:
        payload = json.dumps(
            {
                "mood": "anxious",
                "stability": "strained",
                "risk_level": "moderate",
                "categories": [],
                "confidence": 0.8,
                "rationale": "Distressed but coherent and collaborative.",
            }
        )
        with patch("digital_doctor.safety.risk.call_model", return_value=payload):
            assessment = assess_patient_state("My SUDS is high, but I can keep talking.")

        self.assertTrue(assessment.treatment_allowed)
        self.assertFalse(assessment.notify_human)
        self.assertFalse(assessment.stop_conversation)

    def test_final_review_ignores_non_actionable_style_rewrite(self) -> None:
        reviewer_payload = json.dumps(
            {
                "risk_level": "low",
                "categories": ["emotional_instability", "other"],
                "action": "revise",
                "confidence": 0.8,
                "rationale": "Patient sounds anxious.",
                "revision_instruction": "Make it more supportive.",
            }
        )
        draft = "That sounds difficult. What happened next?"
        with patch("digital_doctor.safety.review.call_model", return_value=reviewer_payload):
            verdict = review_final_response(
                "I'm anxious.",
                draft,
                treatment_allowed=False,
                mood_assessment={"stability": "strained", "risk_level": "moderate"},
            )

        self.assertEqual(verdict.action, "allow")
        self.assertFalse(verdict.changed)
        self.assertEqual(verdict.final_reply, draft)

    def test_final_review_still_revises_explicit_reassurance(self) -> None:
        reviewer_payload = json.dumps(
            {
                "risk_level": "low",
                "categories": ["reassurance_given"],
                "action": "revise",
                "confidence": 0.9,
                "rationale": "Draft promises safety.",
                "revision_instruction": "Remove certainty about the feared outcome.",
            }
        )
        with patch(
            "digital_doctor.safety.review.call_model",
            side_effect=[reviewer_payload, "We cannot know for certain what will happen."],
        ):
            verdict = review_final_response(
                "Will I get sick?",
                "You definitely will not get sick.",
                treatment_allowed=False,
                mood_assessment={"stability": "stable", "risk_level": "low"},
            )

        self.assertEqual(verdict.action, "revise")
        self.assertTrue(verdict.changed)
        self.assertNotIn("definitely", verdict.final_reply)

    def test_final_review_rejects_copied_category_menu(self) -> None:
        reviewer_payload = json.dumps(
            {
                "risk_level": "low",
                "categories": [
                    "self_harm",
                    "harm_to_others_intent",
                    "abuse_or_safeguarding",
                    "acute_psychiatric",
                    "out_of_scope",
                    "reassurance_given",
                    "unsafe_exposure",
                    "medical_or_medication_advice",
                ],
                "action": "revise",
                "confidence": 0.8,
                "rationale": "Copied category menu.",
                "revision_instruction": "Rewrite it.",
            }
        )
        draft = "That sounds difficult. What happened next?"
        with patch("digital_doctor.safety.review.call_model", return_value=reviewer_payload):
            verdict = review_final_response(
                "I'm anxious.",
                draft,
                treatment_allowed=False,
                mood_assessment={"stability": "strained", "risk_level": "moderate"},
            )

        self.assertEqual(verdict.action, "allow")
        self.assertEqual(verdict.final_reply, draft)

    def test_final_review_fails_closed_when_model_is_unavailable(self) -> None:
        with patch("digital_doctor.safety.review.call_model", return_value="not-json"):
            advice = review_final_response(
                "What should I do?",
                "Try an exposure exercise and response prevention tonight.",
                treatment_allowed=True,
            )
            obsession = review_final_response(
                "I have an unwanted intrusive thought that I might hurt myself, but I don't want to.",
                "Thank you for explaining that the thought is unwanted.",
                treatment_allowed=False,
            )
        self.assertEqual(advice.action, "escalate")
        self.assertNotIn("exposure exercise", advice.final_reply)
        self.assertNotEqual(obsession.action, "crisis")

    def test_notifier_writes_durable_alert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            alert_path = Path(tmpdir) / "alerts.jsonl"
            delivered = []
            notifier = ClinicalAlertNotifier(
                path=str(alert_path), delivery_callback=lambda record: delivered.append(record["alert_id"])
            )
            assessment = MoodAssessment(
                mood="overwhelmed",
                stability="critical",
                risk_level="critical",
                treatment_allowed=False,
                notify_human=True,
                stop_conversation=True,
            )
            alert = notifier.notify(assessment, "I cannot stay safe.", "u", "e", "s")
            persisted = json.loads(alert_path.read_text(encoding="utf-8").strip())
        self.assertEqual(alert["status"], "delivered_callback")
        self.assertEqual(persisted["alert_id"], alert["alert_id"])
        self.assertEqual(delivered, [alert["alert_id"]])

    def test_critical_risk_stops_session_and_queues_alert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcript = root / "transcript.json"
            transcript.write_text(json.dumps({"Transcripts": []}), encoding="utf-8")
            milestone = Milestone(1, "ERP rationale", "Explain ERP.", [])
            critical = MoodAssessment(
                mood="overwhelmed",
                stability="critical",
                risk_level="critical",
                categories=["self_harm"],
                confidence=0.99,
                rationale="Explicit plan and intent.",
                treatment_allowed=False,
                notify_human=True,
                stop_conversation=True,
            )
            with patch("digital_doctor.core.session.load_milestones", return_value=[milestone]):
                session = DigitalDoctorSession(
                    transcript_path=str(transcript),
                    milestone_path=str(root / "unused.md"),
                    memory_path=str(root / "memory.jsonl"),
                    long_term_memory_path=str(root / "long_term.json"),
                    state_path=str(root / "state.jsonl"),
                    log_path=str(root / "debug.log"),
                    trace_path=str(root / "trace.jsonl"),
                    alert_path=str(root / "alerts.jsonl"),
                    use_helper_model=False,
                    use_knowledge_tree=False,
                )
            with patch("digital_doctor.core.session.assess_patient_state", return_value=critical):
                reply, update = session.handle_query("I have a plan to kill myself tonight.")
            second_reply, second_update = session.handle_query("Can we continue ERP?")

            with patch("digital_doctor.core.session.load_milestones", return_value=[milestone]):
                restored = DigitalDoctorSession(
                    transcript_path=str(transcript),
                    milestone_path=str(root / "unused.md"),
                    memory_path=str(root / "memory.jsonl"),
                    long_term_memory_path=str(root / "long_term.json"),
                    state_path=str(root / "state.jsonl"),
                    log_path=str(root / "debug.log"),
                    trace_path=str(root / "trace.jsonl"),
                    alert_path=str(root / "alerts.jsonl"),
                    use_helper_model=False,
                    use_knowledge_tree=False,
                )

            self.assertIn("stopping the treatment discussion", reply)
            self.assertTrue(update["conversation_stopped"])
            self.assertEqual(update["safety"]["action"], "crisis")
            self.assertTrue((root / "alerts.jsonl").exists())
            self.assertTrue(second_update["conversation_stopped"])
            self.assertIn("treatment conversation is paused", second_reply)
            self.assertTrue(restored.conversation_stopped)


if __name__ == "__main__":
    unittest.main()
