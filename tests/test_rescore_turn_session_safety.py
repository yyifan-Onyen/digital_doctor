from __future__ import annotations

import unittest

from digital_doctor.tools.rescore_turn_session_safety import (
    _map_turn_result,
    render_report,
)


class TurnSessionSafetyReportTests(unittest.TestCase):
    def test_maps_blinded_turn_fields_back_to_systems(self) -> None:
        item = {
            "patient_state": "distressed",
            "acceptable_intent": "assess",
            "maximum_action": "one question",
            "expected_state_delta": "clarify trigger",
            "turn_score_a": 5,
            "turn_score_b": 2,
            "clinically_acceptable_a": True,
            "clinically_acceptable_b": False,
            "safety_level_a": "none",
            "safety_level_b": "major",
            "premature_erp_a": False,
            "premature_erp_b": True,
            "reassurance_violation_a": False,
            "reassurance_violation_b": False,
            "reason_a": "Appropriate assessment.",
            "reason_b": "Premature treatment.",
        }
        mapped = _map_turn_result(item, {"A": "harness", "B": "raw"})
        self.assertEqual(mapped["harness"]["turn_score"], 5)
        self.assertEqual(mapped["raw"]["turn_score"], 2)
        self.assertEqual(mapped["raw"]["safety_level"], "major")

    def test_report_has_only_session_turn_and_safety_score_layers(self) -> None:
        system = {
            "dimensions": [
                {
                    "id": index,
                    "name": f"Dimension {index}",
                    "score_mean": 4,
                    "score_passes": [4, 4],
                    "evidence_turn_ids": ["C01"],
                    "reasons": ["Reason", "Reason"],
                }
                for index in range(1, 11)
            ],
            "overall_session_rating_mean": 4,
            "overall_session_rating_passes": [4, 4],
            "clinically_acceptable_votes": [True, True],
            "critical_failure_counts": [0, 0],
            "major_violation_counts": [0, 0],
            "summaries": ["Summary", "Summary"],
        }
        turn_system = {
            "turn_score_mean": 4,
            "turn_score_passes": [4, 4],
            "clinically_acceptable": True,
            "acceptability_votes": [True, True],
            "safety_level": "none",
            "safety_level_passes": ["none", "none"],
            "premature_erp": False,
            "reassurance_violation": False,
            "reasons": ["Reason", "Reason"],
        }
        row = {
            "checkpoint_id": "C01",
            "reference_phase": "Assessment",
            "turn_evaluation_v02": {
                "raw": turn_system,
                "harness": turn_system,
            },
        }
        session = {"systems": {"raw": system, "harness": system}}
        report = render_report([row], session, "zh")
        self.assertIn("Layer 1 · Session", report)
        self.assertIn("Layer 2 · Turn", report)
        self.assertIn("Layer 3 · Safety", report)
        self.assertIn("Clinical Evaluation Overview", report)
        self.assertIn("QUALITY SCORES", report)
        self.assertIn("SAFETY FLAGS", report)
        self.assertIn("Session-level competency radar", report)
        self.assertIn("Turn-level performance by clinical phase", report)
        self.assertIn("Safety event profile", report)
        self.assertEqual(report.count("<figure "), 4)
        self.assertNotIn("逐 Turn 平均分", report)
        self.assertNotIn("API 调用", report)
        self.assertNotIn("估算成本", report)


if __name__ == "__main__":
    unittest.main()
