from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from digital_doctor.core.session import DigitalDoctorSession
from digital_doctor.core.session_logic import decide_route, response_move_instructions
from digital_doctor.safety.risk import MoodAssessment


STABLE = MoodAssessment(
    mood="calm",
    stability="stable",
    risk_level="low",
    confidence=0.9,
    treatment_allowed=True,
)


class ConversationRoutingTests(unittest.TestCase):
    def test_router_receives_current_and_next_phase_steering(self) -> None:
        prompts = []

        def route_response(prompt_text: str, json_mode: bool = False, model: str = "") -> str:
            prompts.append(prompt_text)
            return json.dumps(
                {
                    "mode": "analysis",
                    "response_move": "build_buy_in",
                    "depth": "standard",
                    "reason": "current buy-in gap",
                }
            )

        milestone_context = (
            "Current priority phase: P3: ERP Buy-In\n"
            "Current phase exit criteria: tentative willingness\n"
            "Next phase after completion: P4: Exposure Hierarchy"
        )
        with patch("digital_doctor.core.session_logic.call_model", side_effect=route_response):
            route = decide_route(
                "I think I understand why the washing keeps the fear going.",
                "doctor: We discussed short-term relief and long-term learning.",
                lambda _event, _payload: None,
                milestone_context=milestone_context,
            )

        self.assertEqual(route["response_move"], "build_buy_in")
        self.assertIn(milestone_context, prompts[0])
        self.assertIn("naturally opens the next phase", prompts[0])

    def test_router_keeps_clinical_backchannel_in_analysis_with_acknowledgment(self) -> None:
        traces = []
        with patch(
            "digital_doctor.core.session_logic.call_model",
            return_value=json.dumps(
                {
                    "mode": "analysis",
                    "reason": "brief response inside clinical exchange",
                }
            ),
        ):
            route = decide_route(
                "Mm-hm.",
                "doctor: What happens after you wash?\npatient: I feel relief.",
                lambda event, payload: traces.append((event, payload)),
            )

        self.assertEqual(route["mode"], "analysis")
        self.assertEqual(route["response_move"], "acknowledge")
        self.assertEqual(route["depth"], "brief")
        self.assertEqual(traces[-1][1]["response_move"], "acknowledge")

    def test_router_preserves_explicit_natural_clinical_move(self) -> None:
        with patch(
            "digital_doctor.core.session_logic.call_model",
            return_value=json.dumps(
                {
                    "mode": "analysis",
                    "response_move": "reflect",
                    "depth": "brief",
                    "reason": "patient is describing impact",
                }
            ),
        ):
            route = decide_route(
                "My hands are red and cracked from washing.",
                "",
                lambda _event, _payload: None,
            )

        self.assertEqual(route["mode"], "analysis")
        self.assertEqual(route["response_move"], "reflect")
        self.assertIn("continue the unfinished", response_move_instructions({
            "response_move": "acknowledge",
            "depth": "brief",
        }))

    def test_invalid_router_output_falls_back_to_casual_greeting(self) -> None:
        with patch("digital_doctor.core.session_logic.call_model", return_value="not-json"):
            route = decide_route("Hello", "", lambda _event, _payload: None)

        self.assertEqual(route["mode"], "chat")
        self.assertEqual(route["response_move"], "casual")

    def test_unknown_short_token_stays_in_clinical_context(self) -> None:
        with patch(
            "digital_doctor.core.session_logic.call_model",
            return_value=json.dumps(
                {
                    "mode": "chat",
                    "response_move": "casual",
                    "depth": "brief",
                    "reason": "unknown token",
                }
            ),
        ):
            route = decide_route(
                "Dev.",
                "patient: I wash for 30 minutes.\ndoctor: That sounds painful and disruptive.",
                lambda _event, _payload: None,
            )

        self.assertEqual(route["mode"], "analysis")
        self.assertEqual(route["response_move"], "acknowledge")

    def test_forgetting_cue_selects_psychoeducation_move(self) -> None:
        with patch(
            "digital_doctor.core.session_logic.call_model",
            return_value=json.dumps(
                {
                    "mode": "analysis",
                    "response_move": "clarify",
                    "depth": "standard",
                    "reason": "patient is uncertain",
                }
            ),
        ):
            route = decide_route(
                "I honestly can't remember exactly why compulsions are unhelpful.",
                "doctor: We were mapping the washing cycle.",
                lambda _event, _payload: None,
            )

        self.assertEqual(route["response_move"], "psychoeducation")

    def test_chat_route_skips_formulation_extraction_and_phase_progression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fixture = Path(__file__).parent / "fixtures" / "milestones.md"
            with patch.dict("os.environ", {"USE_SAFETY_CHECK": "0"}):
                session = DigitalDoctorSession(
                    transcript_path=str(root / "missing.json"),
                    milestone_path=str(fixture),
                    user_id="u",
                    episode_id="e",
                    patient_role_label="patient",
                    doctor_role_label="doctor",
                    memory_path=str(root / "memory.jsonl"),
                    long_term_memory_path=str(root / "long_term.json"),
                    state_path=str(root / "state.jsonl"),
                    log_path=str(root / "debug.log"),
                    trace_path=str(root / "trace.jsonl"),
                    alert_path=str(root / "alerts.jsonl"),
                    use_helper_model=False,
                    use_knowledge_tree=False,
                )

            with (
                patch("digital_doctor.core.session.assess_patient_state", return_value=STABLE),
                patch.object(
                    session,
                    "_decide_route",
                    return_value={
                        "mode": "chat",
                        "response_move": "casual",
                        "depth": "brief",
                        "reason": "greeting",
                    },
                ),
                patch("digital_doctor.core.session.generate_chat_reply", return_value="Good morning."),
                patch.object(session.tracker, "observe_user_turn") as observe,
            ):
                reply, update = session.handle_query("Good morning")

            self.assertEqual(reply, "Good morning.")
            observe.assert_not_called()
            self.assertEqual(update["route"], "chat")
            self.assertTrue(update["skipped_for_chat"])
            self.assertEqual(session.tracker.turn_idx, 0)


if __name__ == "__main__":
    unittest.main()
