from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from digital_doctor.core.session import DigitalDoctorSession
from digital_doctor.harness.adapters import HttpModelAdapter, OPSDModelAdapter, SFTModelAdapter
from digital_doctor.harness.contracts import ActionPlan, SkillManifest
from digital_doctor.harness.platform_safety import PlatformSafetyGate
from digital_doctor.harness.registry import SkillNotFoundError, SkillRegistry
from digital_doctor.harness.training import export_distillation_records
from digital_doctor.safety.risk import MoodAssessment
from digital_doctor.skills.base import ClinicalSkill
from digital_doctor.skills.ocd_erp.skill import OcdErpSkill


STABLE = MoodAssessment(
    mood="calm",
    stability="stable",
    risk_level="low",
    confidence=0.9,
    treatment_allowed=True,
)


class _SkillStub:
    def __init__(self, version: str) -> None:
        self.manifest = SkillManifest(
            skill_id="stub",
            name="Stub",
            version=version,
            description="test",
            entry_point="tests:_SkillStub",
        )


class _ClientStub:
    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        return prompt


class SkillRegistryTests(unittest.TestCase):
    def test_registry_pins_exact_version_and_resolves_latest(self) -> None:
        registry = SkillRegistry([_SkillStub("1.0.0"), _SkillStub("1.1.0")])
        self.assertEqual(registry.resolve("stub").manifest.version, "1.1.0")
        self.assertEqual(registry.resolve("stub", "1.0.0").manifest.version, "1.0.0")
        with self.assertRaises(SkillNotFoundError):
            registry.resolve("missing")

    def test_learned_model_adapters_have_distinct_trace_identities(self) -> None:
        client = _ClientStub()
        self.assertEqual(HttpModelAdapter(client).adapter_id, "http-model")
        self.assertEqual(SFTModelAdapter(client).adapter_id, "sft-model")
        self.assertEqual(OPSDModelAdapter(client).adapter_id, "opsd-model")

    def test_ocd_erp_bundle_implements_the_complete_skill_contract(self) -> None:
        skill = OcdErpSkill()
        self.assertIsInstance(skill, ClinicalSkill)
        self.assertEqual(skill.manifest.skill_id, "ocd_erp")
        self.assertEqual(len(skill.manifest.bundle_checksum), 64)


class PlatformSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = PlatformSafetyGate(lambda: "stopped", lambda: "critical")
        self.plan = ActionPlan(
            mode="analysis",
            action="treatment_step",
            depth="standard",
            reason="test",
            allowed_actions=["treatment_step"],
            authorized=False,
        )

    def test_final_gate_fails_closed_without_safe_output_proof(self) -> None:
        decision = self.gate.finalize("Proceed with exposure.", STABLE, self.plan, {})
        self.assertTrue(decision.changed)
        self.assertEqual(decision.reason, "unauthorized_action_not_proven_removed")
        self.assertNotIn("exposure", decision.reply.lower())

    def test_final_gate_accepts_proven_removal_of_unauthorized_advice(self) -> None:
        reply = "Let's first understand what has been happening."
        decision = self.gate.finalize(
            reply,
            STABLE,
            self.plan,
            {"final_check": {"advice_detected": False}},
        )
        self.assertFalse(decision.changed)
        self.assertEqual(decision.reply, reply)


class HarnessTraceTests(unittest.TestCase):
    def test_session_pins_skill_and_emits_distillation_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcript = root / "transcript.json"
            transcript.write_text(json.dumps({"Transcripts": []}), encoding="utf-8")
            trace_path = root / "trace.jsonl"
            with patch.dict("os.environ", {"USE_SAFETY_CHECK": "0"}):
                session = DigitalDoctorSession(
                    transcript_path=str(transcript),
                    milestone_path=str(Path(__file__).parent / "fixtures" / "milestones.md"),
                    memory_path=str(root / "memory.jsonl"),
                    long_term_memory_path=str(root / "long-term.json"),
                    state_path=str(root / "state.jsonl"),
                    log_path=str(root / "debug.log"),
                    trace_path=str(trace_path),
                    alert_path=str(root / "alerts.jsonl"),
                    single_turn=True,
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
                patch("digital_doctor.core.session.generate_chat_reply", return_value="Hello."),
            ):
                reply, update = session.handle_query("Hello")

            events = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            distillation = [item for item in events if item["event"] == "distillation_record"]

            self.assertEqual(reply, "Hello.")
            self.assertEqual(update["harness"]["skill_id"], "ocd_erp")
            self.assertEqual(update["action_plan"]["action"], "casual")
            self.assertEqual(len(update["harness"]["skill_checksum"]), 64)
            self.assertEqual(len(distillation), 1)
            self.assertEqual(
                distillation[0]["teacher_target"]["response"],
                "Hello.",
            )

    def test_trace_export_supports_sft_and_opsd(self) -> None:
        event = {
            "event": "distillation_record",
            "identity": {"skill_id": "ocd_erp", "skill_version": "1.0.0"},
            "student_input": {"history": "", "patient_message": "Hello"},
            "privileged_skill_context": {"action_plan": {"action": "casual"}},
            "teacher_target": {"response": "Hello.", "safety": {}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace = root / "trace.jsonl"
            trace.write_text(json.dumps(event) + "\n", encoding="utf-8")
            sft = root / "sft.jsonl"
            opsd = root / "opsd.jsonl"
            self.assertEqual(export_distillation_records(trace, sft, "sft"), 1)
            self.assertEqual(export_distillation_records(trace, opsd, "opsd"), 1)
            sft_record = json.loads(sft.read_text(encoding="utf-8"))
            opsd_record = json.loads(opsd.read_text(encoding="utf-8"))

        self.assertEqual(sft_record["messages"][1]["content"], "Hello.")
        self.assertIn("privileged_teacher_context", opsd_record)


if __name__ == "__main__":
    unittest.main()
