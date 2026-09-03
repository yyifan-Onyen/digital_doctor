from __future__ import annotations

import unittest

from digital_doctor.tools.harness_workflow_eval import (
    _map_pairwise_result,
    _oriented_cases,
    build_raw_prompt,
    response_metrics,
    summarize_calls,
)


class HarnessWorkflowEvaluationTests(unittest.TestCase):
    def test_raw_prompt_uses_shared_history_and_latest_patient_message(self) -> None:
        prompt = build_raw_prompt("doctor: Hello\npatient: Hi", "I keep checking.")
        self.assertIn("doctor: Hello", prompt)
        self.assertIn("I keep checking.", prompt)
        self.assertNotIn("rubric", prompt.lower())

    def test_cost_summary_separates_cached_input(self) -> None:
        result = summarize_calls(
            [
                {
                    "requested_model": "gpt-5.4-mini-2026-03-17",
                    "served_model": "gpt-5.4-mini-2026-03-17",
                    "input_tokens": 1_000_000,
                    "cached_input_tokens": 200_000,
                    "output_tokens": 100_000,
                    "reasoning_tokens": 10_000,
                    "total_tokens": 1_100_000,
                    "latency_seconds": 1.25,
                    "error_type": "",
                }
            ]
        )
        self.assertEqual(result["call_count"], 1)
        self.assertEqual(result["estimated_cost_usd"], 1.065)

    def test_response_metrics_detect_multiple_questions(self) -> None:
        result = response_metrics(["What happens first? And then what?", "Okay."])
        self.assertEqual(result["response_count"], 2)
        self.assertEqual(result["multi_question_count"], 1)
        self.assertEqual(result["multi_question_rate_percent"], 50.0)

    def test_position_swap_maps_results_back_to_system(self) -> None:
        row = {
            "checkpoint_id": "C01",
            "reference_phase": "Assessment",
            "history": "patient: hello",
            "patient_text": "hello",
            "reference_therapist": "Hi.",
            "raw_therapist": "Raw",
            "harness_therapist": "Harness",
        }
        first_cases, first_mapping = _oriented_cases([row], 0)
        second_cases, second_mapping = _oriented_cases([row], 1)
        self.assertEqual(first_cases[0]["response_a"], "Raw")
        self.assertEqual(second_cases[0]["response_a"], "Harness")

        judged = {
            "preferred": "A",
            "clinically_acceptable_a": True,
            "clinically_acceptable_b": False,
            "phase_appropriateness_a": 2,
            "phase_appropriateness_b": 1,
            "critical_failure_a": False,
            "critical_failure_b": False,
            "major_violation_a": False,
            "major_violation_b": True,
            "premature_erp_a": False,
            "premature_erp_b": True,
            "reassurance_violation_a": False,
            "reassurance_violation_b": False,
            "safety_flags_a": [],
            "safety_flags_b": ["premature ERP"],
            "reason": "A is better calibrated.",
        }
        mapped = _map_pairwise_result(judged, first_mapping["C01"])
        self.assertEqual(mapped["preferred_system"], "raw")
        self.assertTrue(mapped["raw"]["clinically_acceptable"])
        self.assertTrue(mapped["harness"]["major_violation"])
        self.assertEqual(second_mapping["C01"]["A"], "harness")


if __name__ == "__main__":
    unittest.main()
