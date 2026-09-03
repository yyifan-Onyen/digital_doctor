from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from digital_doctor.core.session import DigitalDoctorSession
from digital_doctor.safety.risk import MoodAssessment
from digital_doctor.tracking.milestones import MilestoneTracker, load_milestones


def _tracking_response(prompt_text: str, json_mode: bool = False, model: str = "") -> str:
    if "updating a structured OCD case formulation" in prompt_text:
        return json.dumps(
            {
                "updates": [
                    {
                        "field": "obsession",
                        "value": "contamination fear after touching public surfaces",
                        "confidence": 0.92,
                        "evidence": "felt contaminated after touching the doorknob",
                    },
                    {
                        "field": "trigger",
                        "value": "touching the office doorknob",
                        "confidence": 0.89,
                        "evidence": "after touching the office doorknob",
                    },
                    {
                        "field": "feared_consequence",
                        "value": "bringing germs home to family",
                        "confidence": 0.9,
                        "evidence": "I thought I would contaminate my family",
                    },
                    {
                        "field": "compulsion",
                        "value": "repeated handwashing",
                        "confidence": 0.94,
                        "evidence": "I washed twice right away",
                    },
                    {
                        "field": "stuck_points",
                        "value": "still treats washing as the safest option",
                        "confidence": 0.77,
                        "evidence": "I felt like I had to wash",
                    },
                ]
            }
        )
    if "evaluating phase progress for one ERP therapy turn" in prompt_text:
        return json.dumps(
            {
                "phases": [
                    {
                        "id": 1,
                        "status": "completed",
                        "confidence": 0.94,
                        "evidence": "The turn identifies obsession, trigger, and ritual pattern.",
                        "blocked_reason": "",
                        "contraindication_reason": "",
                    },
                    {
                        "id": 2,
                        "status": "completed",
                        "confidence": 0.88,
                        "evidence": "The case is summarized as trigger-fear-ritual loop.",
                        "blocked_reason": "",
                        "contraindication_reason": "",
                    },
                    {
                        "id": 3,
                        "status": "active",
                        "confidence": 0.72,
                        "evidence": "The clinician explains ERP learning target and asks about willingness.",
                        "blocked_reason": "",
                        "contraindication_reason": "",
                    },
                    {"id": 4, "status": "pending", "confidence": 0.1, "evidence": "", "blocked_reason": "", "contraindication_reason": ""},
                    {"id": 5, "status": "pending", "confidence": 0.1, "evidence": "", "blocked_reason": "", "contraindication_reason": ""},
                    {"id": 6, "status": "pending", "confidence": 0.1, "evidence": "", "blocked_reason": "", "contraindication_reason": ""},
                    {"id": 7, "status": "pending", "confidence": 0.1, "evidence": "", "blocked_reason": "", "contraindication_reason": ""},
                ]
            }
        )
    raise AssertionError(f"Unexpected tracking prompt: {prompt_text[:120]}")


def _session_logic_response(prompt_text: str, json_mode: bool = False, model: str = "") -> str:
    if "Classify the latest user message into one mode" in prompt_text:
        return json.dumps({"mode": "analysis", "reason": "ocd-content"})
    if "Write a content-only draft" in prompt_text:
        return (
            "It makes sense that the doorknob set off the contamination alarm again. "
            "That pull to wash is part of the OCD loop, not proof that danger is present. "
            "Would you be willing to test a brief delay before washing next time?"
        )
    if "Rewrite the draft to align with the cadence" in prompt_text:
        return (
            "That doorknob clearly set off the contamination alarm again, and the urge to wash makes sense in that loop. "
            "The key thing we want to learn is what happens if you make a little more room before responding to OCD. "
            "Would you be open to trying a short delay before washing the next time this happens?"
        )
    raise AssertionError(f"Unexpected session prompt: {prompt_text[:120]}")


class PhasePlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.milestone_path = str(Path(__file__).parent / "fixtures" / "milestones.md")
        self.milestones = load_milestones(self.milestone_path)

    def test_formulation_updates_are_applied_before_phase_progress(self) -> None:
        tracker = MilestoneTracker(self.milestones, session_goal="Progress through ERP phases.")
        with patch("digital_doctor.tracking.milestones.call_model", side_effect=_tracking_response):
            observation = tracker.observe_user_turn(
                "I touched the office doorknob and washed twice because I thought I would contaminate my family.",
                history="",
            )

            self.assertEqual(observation["formulation_filled_count"], 5)
            self.assertEqual(
                tracker.formulation.fields["obsession"].value,
                "contamination fear after touching public surfaces",
            )

            update = tracker.update(
                "I touched the office doorknob and washed twice because I thought I would contaminate my family.",
                "The washing urge sounds like the OCD loop. Would you be willing to test a brief delay before washing?",
            )

        self.assertEqual(update["covered_count"], 2)
        self.assertEqual(update["next_target"], "P3: ERP Buy-In")
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["formulation"]["filled_count"], 5)
        self.assertEqual(snapshot["phases"][0]["status"], "completed")
        self.assertEqual(snapshot["phases"][1]["status"], "completed")
        self.assertEqual(snapshot["phases"][2]["status"], "active")

    def test_planner_context_exposes_exit_criteria_and_next_phase_bridge(self) -> None:
        tracker = MilestoneTracker(self.milestones, session_goal="Progress through ERP phases.")

        context = tracker.render_context()

        self.assertIn("Current priority phase: P1: Assessment", context)
        self.assertIn("Current phase exit criteria:", context)
        self.assertIn("Next phase after completion: P2: Formulation", context)
        self.assertIn("address the smallest unmet current-phase criterion", context)
        self.assertIn("Never announce phase labels to the patient", context)

    def test_planner_emits_diagnostics_transitions_and_healthy_status(self) -> None:
        events = []
        tracker = MilestoneTracker(
            self.milestones,
            session_goal="Progress through ERP phases.",
            event_writer=lambda event, payload: events.append((event, payload)),
        )
        with patch("digital_doctor.tracking.milestones.call_model", side_effect=_tracking_response):
            tracker.observe_user_turn(
                "I touched the office doorknob and washed twice because I thought I would contaminate my family."
            )
            update = tracker.update(
                "I touched the office doorknob and washed twice because I thought I would contaminate my family.",
                "That gives us a clear trigger-fear-washing loop.",
            )

        event_names = [event for event, _payload in events]
        self.assertIn("milestone_formulation_inference", event_names)
        self.assertIn("milestone_formulation_updated", event_names)
        self.assertIn("milestone_phase_inference", event_names)
        self.assertIn("milestone_state_transition", event_names)
        self.assertIn("milestone_health", event_names)
        self.assertEqual(update["milestone_health"]["status"], "healthy")
        self.assertTrue(update["milestone_health"]["phase_inference"]["output_complete"])
        self.assertTrue(update["transition"]["advanced"])
        self.assertEqual(update["transition"]["from_phase_id"], 1)
        self.assertEqual(update["transition"]["to_phase_id"], 3)

    def test_planner_health_reports_malformed_phase_output(self) -> None:
        events = []
        tracker = MilestoneTracker(
            self.milestones,
            session_goal="Progress through ERP phases.",
            event_writer=lambda event, payload: events.append((event, payload)),
        )

        with patch("digital_doctor.tracking.milestones.call_model", return_value="not-json"):
            update = tracker.update("I feel stuck.", "Let's understand what is getting in the way.")

        self.assertEqual(update["milestone_health"]["status"], "degraded")
        self.assertFalse(update["milestone_health"]["phase_inference"]["parse_ok"])
        self.assertEqual(update["current_phase"], "P1: Assessment")
        inference_event = next(
            payload for event, payload in events if event == "milestone_phase_inference"
        )
        self.assertIn("JSONDecodeError", inference_event["parse_error"])

    def test_structured_phase_floors_prevent_llm_from_stalling_completed_work(self) -> None:
        tracker = MilestoneTracker(self.milestones, session_goal="Progress through ERP phases.")
        for field, value in {
            "obsession": "contamination fear",
            "trigger": "bathroom doorknob",
            "feared_consequence": "getting sick and missing work",
            "compulsion": "washing and Lysol",
            "avoidance": "avoids the bathroom",
        }.items():
            tracker.formulation.fields[field].value = value
        tracker.turn_idx = 7
        all_active = [
            {
                "id": phase.phase_id,
                "status": "active" if phase.phase_id == 1 else "pending",
                "confidence": 0.9,
                "evidence": "model remained conservative",
                "blocked_reason": "",
                "contraindication_reason": "",
            }
            for phase in tracker.phases
        ]

        with patch.object(tracker, "_infer_phase_progress", return_value=all_active):
            tracker.update("The washing gives brief relief.", "That maps the maintaining loop.")
            self.assertEqual(tracker.state[1].status, "completed")
            self.assertEqual(tracker.state[2].status, "completed")
            self.assertEqual(tracker.state[3].status, "active")

            tracker.update(
                "I'm not sure yet.",
                "Would you be willing to try if we understood the rationale together?",
            )
            self.assertEqual(tracker.state[3].status, "active")

            tracker.update(
                "That makes sense—some short-term discomfort could help in the longer term.",
                "Exactly.",
            )

        self.assertEqual(tracker.state[3].status, "completed")
        self.assertEqual(tracker.state[4].status, "active")

    def test_session_handle_query_uses_phase_planner_and_formulation_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcript_path = root / "transcript.json"
            transcript_path.write_text(json.dumps({"Transcripts": []}), encoding="utf-8")

            session = DigitalDoctorSession(
                transcript_path=str(transcript_path),
                milestone_path=self.milestone_path,
                memory_path=str(root / "memory.jsonl"),
                state_path=str(root / "state.jsonl"),
                log_path=str(root / "debug.log"),
                trace_path=str(root / "trace.jsonl"),
                single_turn=True,
                use_helper_model=False,
                use_knowledge_tree=False,
            )
            session.use_safety_check = False

            with patch("digital_doctor.tracking.milestones.call_model", side_effect=_tracking_response), patch(
                "digital_doctor.core.session_logic.call_model",
                side_effect=_session_logic_response,
            ), patch(
                "digital_doctor.core.session.assess_patient_state",
                return_value=MoodAssessment(
                    mood="calm",
                    stability="stable",
                    risk_level="low",
                    confidence=0.9,
                    treatment_allowed=True,
                ),
            ), patch("digital_doctor.core.session.pick_refs", return_value=[]):
                reply, update = session.handle_query(
                    "I touched the office doorknob and washed twice because I thought I would contaminate my family."
                )

            trace_events = [
                json.loads(line)["event"]
                for line in (root / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            debug_log = (root / "debug.log").read_text(encoding="utf-8")

        self.assertNotIn("short delay before washing", reply)
        self.assertIn("Before we decide on any treatment step", reply)
        self.assertEqual(update["route"], "analysis")
        self.assertEqual(update["current_phase"], "P3: ERP Buy-In")
        self.assertEqual(update["formulation_filled_count"], 5)
        self.assertTrue(update["formulation_updates"])
        self.assertFalse(update["treatment_readiness"]["allowed"])
        self.assertEqual(update["treatment_readiness"]["stage"], "collecting_context")
        self.assertTrue(update["treatment_output_gate"]["before_safety"]["changed"])
        artifacts = update["artifacts"]
        self.assertIn("pre_turn_observation", artifacts)
        self.assertEqual(artifacts["pre_turn_observation"]["formulation_filled_count"], 5)
        self.assertEqual(artifacts["milestone_snapshot_before_turn"]["formulation"]["filled_count"], 5)
        self.assertIn("milestone_phase_inference", trace_events)
        self.assertIn("milestone_state_transition", trace_events)
        self.assertIn("milestone_health", trace_events)
        self.assertIn("[milestone] turn=1 route=analysis health=healthy", debug_log)


if __name__ == "__main__":
    unittest.main()
