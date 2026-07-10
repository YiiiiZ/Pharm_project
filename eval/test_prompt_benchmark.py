from unittest import TestCase

from care.care_plan_schema import CarePlan
from eval.prompt_benchmark import (
    aggregate_results,
    assess_comparison_gates,
    build_reference_material,
    compare_runs,
    load_benchmark_cases,
    score_plan,
)


def minimal_plan(case) -> CarePlan:
    patient = case["input"]["patient"]
    order = case["input"]["order"]
    first_problem = case["golden"]["problem_list"][0]
    first_goal = case["golden"]["goals"][0]
    first_intervention = case["golden"]["interventions"][0]
    return CarePlan.model_validate(
        {
            "patient_summary": {
                "patient_name": (
                    f"{patient['first_name']} {patient['last_name']}"
                ),
                "mrn": patient["mrn"],
                "date_of_birth": patient.get("dob"),
                "weight_kg": order.get("weight_kg"),
                "allergies": [order.get("allergies", "None")],
                "medication_name": order["medication_name"],
                "primary_diagnosis": order["primary_diagnosis"],
            },
            "problems": [
                {
                    "title": first_problem,
                    "description": first_problem,
                }
            ],
            "goals": [
                {
                    "description": first_goal,
                    "target": first_goal,
                }
            ],
            "interventions": [
                {
                    "action": first_intervention,
                    "rationale": first_intervention,
                }
            ],
        }
    )


class PromptBenchmarkTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.config, cls.cases, cls.test_set_hash = load_benchmark_cases()

    def test_benchmark_has_five_frozen_cases(self):
        self.assertEqual(len(self.cases), 5)
        self.assertEqual(
            [case["id"] for case in self.cases],
            ["case_001", "case_002", "case_004", "case_008", "case_010"],
        )
        self.assertEqual(len(self.test_set_hash), 64)

    def test_reference_material_has_stable_chunk_ids(self):
        material = build_reference_material(self.cases[0])

        self.assertIn('id="case_001_problem_01"', material)
        self.assertIn('type="monitoring"', material)

    def test_scoring_reports_accuracy_and_coverage(self):
        score = score_plan(self.cases[0], minimal_plan(self.cases[0]))

        self.assertGreater(score["accuracy"], 0)
        self.assertGreater(score["coverage"], 0)
        self.assertLess(score["coverage"], 1)

    def test_aggregate_includes_parse_rate_and_tokens(self):
        results = [
            {
                "parse_success": True,
                "score": {"accuracy": 0.8, "coverage": 0.6, "f1": 0.6857},
                "usage": {"input_tokens": 100, "output_tokens": 40},
            },
            {
                "parse_success": False,
                "score": None,
                "usage": {"input_tokens": 80, "output_tokens": 10},
            },
        ]

        aggregate = aggregate_results(
            results,
            input_price_per_million=1.0,
            output_price_per_million=2.0,
        )

        self.assertEqual(aggregate["parse_success_rate"], 0.5)
        self.assertEqual(aggregate["tokens_total"], 230)
        self.assertEqual(aggregate["accuracy_macro"], 0.8)
        self.assertIsNotNone(aggregate["estimated_cost_usd"])

    def test_comparison_calculates_deltas(self):
        baseline = {
            "prompt_version": "v1",
            "aggregate": {
                "parse_success_rate": 0.8,
                "accuracy_macro": 0.7,
                "coverage_macro": 0.6,
                "f1_macro": 0.64,
                "input_tokens_total": 1000,
                "output_tokens_total": 500,
                "tokens_total": 1500,
                "estimated_cost_usd": None,
            },
        }
        current = {
            "prompt_version": "v2",
            "aggregate": {
                "parse_success_rate": 1.0,
                "accuracy_macro": 0.75,
                "coverage_macro": 0.7,
                "f1_macro": 0.72,
                "input_tokens_total": 1100,
                "output_tokens_total": 450,
                "tokens_total": 1550,
                "estimated_cost_usd": None,
            },
        }

        comparison = compare_runs(current, baseline)

        self.assertAlmostEqual(
            comparison["deltas"]["parse_success_rate"],
            0.2,
        )
        self.assertEqual(comparison["deltas"]["tokens_total"], 50)

    def test_comparison_gate_detects_regression(self):
        comparison = {
            "baseline_tokens_total": 1000,
            "current_tokens_total": 1300,
            "deltas": {
                "parse_success_rate": 0.0,
                "accuracy_macro": -0.08,
                "coverage_macro": -0.01,
            },
        }
        gates = {
            "parse_success_rate_max_regression": 0.0,
            "accuracy_max_regression": 0.05,
            "coverage_max_regression": 0.05,
            "token_total_max_increase_ratio": 0.20,
        }

        result = assess_comparison_gates(comparison, gates)

        self.assertFalse(result["passed"])
        self.assertEqual(len(result["failures"]), 2)
