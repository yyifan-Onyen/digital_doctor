from __future__ import annotations

import unittest

from digital_doctor.tools.gold_prefix_eval import (
    TranscriptTurn,
    assign_roles,
    build_checkpoints,
    reference_phase,
    render_clinical_comparison_en,
    render_clinician_html_report,
    render_html_report,
    render_implementation_report_zh,
)


class GoldPrefixEvaluationTests(unittest.TestCase):
    def test_builds_patient_checkpoints_with_the_original_prefix_count(self) -> None:
        raw = [
            TranscriptTurn("Therapist", "0:01", "How are you?"),
            TranscriptTurn("Patient", "0:02", "Stressed."),
            TranscriptTurn("Therapist", "0:03", "What happened?"),
            TranscriptTurn("Patient", "0:04", "I touched a doorknob."),
            TranscriptTurn("Therapist", "0:05", "What urge came up?"),
        ]
        turns = assign_roles(raw, therapist_hint="Therapist")
        leading, checkpoints = build_checkpoints(turns)

        self.assertEqual(len(leading), 1)
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(checkpoints[0].prefix_turn_count, 1)
        self.assertEqual(checkpoints[1].prefix_turn_count, 3)
        self.assertEqual(checkpoints[1].therapist.text, "What urge came up?")

    def test_reference_phase_windows_cover_requested_milestones(self) -> None:
        self.assertEqual(reference_phase("5:44"), "Assessment")
        self.assertEqual(reference_phase("6:02"), "Formulation")
        self.assertEqual(reference_phase("9:37"), "ERP Buy-In")

    def test_html_escapes_dialogue_content(self) -> None:
        row = {
            "checkpoint_id": "C01",
            "patient_timestamp": "0:02",
            "therapist_timestamp": "0:03",
            "prefix_turn_count": 1,
            "history_chars": 12,
            "reference_phase": "Assessment",
            "patient_text": "<script>alert(1)</script>",
            "reference_therapist": "Tell me more.",
            "digital_therapist": "What happened?",
            "route": "analysis",
            "response_move": "assess",
            "response_depth": "standard",
            "route_reason": "clinical",
            "phase_before": "P1",
            "phase_after": "P1",
            "phase_status_changes": [],
            "formulation_updates": [],
            "mood": "calm",
            "stability": "stable",
            "risk_level": "low",
            "treatment_allowed": False,
            "treatment_stage": "collecting_context",
            "treatment_missing_context": [],
            "guardrail_before": "What happened?",
            "guardrail_after": "What happened?",
            "guardrail_changed": False,
            "safety_action": "allow",
            "safety_categories": [],
            "memory_recall": {},
            "difference_label": "neutral",
            "expected_response_move": "assess",
            "comparison_summary": "Same function.",
            "safety_notes": "",
            "clinical_alignment_score": 4,
            "naturalness_score": 4,
        }
        report = render_html_report([row], __import__("pathlib").Path("test.docx"), "model", "now")

        self.assertNotIn("<script>alert(1)</script>", report)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", report)

        clinician_report = render_clinician_html_report(
            [row], __import__("pathlib").Path("test.docx"), "model", "now"
        )
        self.assertNotIn("<script>alert(1)</script>", clinician_report)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", clinician_report)
        self.assertIn("医生审阅结论", clinician_report)

        english_report = render_clinical_comparison_en(
            [row], __import__("pathlib").Path("test.docx"), "model", "now"
        )
        self.assertNotIn("<script>alert(1)</script>", english_report)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", english_report)
        self.assertIn("<h3>Patient</h3>", english_report)
        self.assertNotIn("Clinical review status", english_report)
        self.assertNotIn("Clinical state, formulation, memory, and guardrail evidence", english_report)
        self.assertNotIn("Evaluation design", english_report)

        implementation_report = render_implementation_report_zh(
            [row], __import__("pathlib").Path("test.docx"), "model", "now"
        )
        self.assertNotIn("<script>alert(1)</script>", implementation_report)
        self.assertIn("具体实现说明", implementation_report)
        self.assertNotIn("代码位置", implementation_report)
        self.assertNotIn("复现与验证", implementation_report)
        self.assertNotIn("当前限制", implementation_report)
        self.assertNotIn("测试结果", implementation_report)
        self.assertNotIn("Response move 分布", implementation_report)
        self.assertNotIn("Guardrail 修改记录", implementation_report)
        self.assertIn("状态更新顺序", implementation_report)
        self.assertIn("每轮执行", implementation_report)
        self.assertIn("Helper Model 协作链路", implementation_report)
        self.assertIn("Milestone 加载与 Phase 映射", implementation_report)


if __name__ == "__main__":
    unittest.main()
