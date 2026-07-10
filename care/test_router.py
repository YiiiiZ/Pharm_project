from unittest import TestCase

from care.router import assess_complexity


class AssessComplexityTests(TestCase):
    def test_simple_case_routes_to_lighter_strategy(self):
        strategy = assess_complexity(
            {
                "diagnoses": ["G70.00"],
                "medications": ["IVIG", "Pyridostigmine"],
                "allergies": [],
                "lab_values": {"SCr": 0.8},
            }
        )

        self.assertEqual(strategy.complexity, "simple")
        self.assertEqual(strategy.prompt_version, "v1")
        self.assertEqual(strategy.model, "gpt-5.5")
        self.assertIn("diagnoses count=1", strategy.decision_log[0])
        self.assertIn("medications count=2", strategy.decision_log[1])
        self.assertIn("allergy history=absent", strategy.decision_log[2])
        self.assertIn("labs provided=True", strategy.decision_log[3])
        self.assertEqual(strategy.decision_log[-1], "decision=simple")

    def test_complex_case_routes_to_stronger_strategy(self):
        strategy = assess_complexity(
            {
                "diagnoses": ["G70.00", "N18.9"],
                "medications": ["IVIG", "Pyridostigmine"],
                "allergies": [],
                "lab_values": {},
            }
        )

        self.assertEqual(strategy.complexity, "complex")
        self.assertEqual(strategy.prompt_version, "v2")
        self.assertEqual(strategy.model, "claude-sonnet-4-6")
        self.assertEqual(strategy.decision_log[-1], "decision=complex")

    def test_allergy_history_forces_complex_route(self):
        strategy = assess_complexity(
            {
                "diagnoses": ["G70.00"],
                "medications": ["IVIG"],
                "allergies": ["Penicillin"],
                "lab_values": {},
            }
        )

        self.assertEqual(strategy.complexity, "complex")
        self.assertIn("allergy history=present", strategy.decision_log[2])

    def test_string_inputs_are_normalized(self):
        strategy = assess_complexity(
            {
                "diagnoses": "G70.00",
                "medications": "IVIG",
                "allergies": "None known",
                "labs": {"SCr": 0.8},
            }
        )

        self.assertEqual(strategy.complexity, "simple")
        self.assertEqual(strategy.prompt_version, "v1")
        self.assertIn("labs provided=True", strategy.decision_log[3])
