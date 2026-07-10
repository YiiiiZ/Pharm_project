from unittest import TestCase

from care.generation_loop import (
    GeneratedOutput,
    HallucinationFinding,
    HallucinationReport,
    LoopStatus,
)
from care.orchestrator import CarePlanOrchestrator


class SequenceGenerator:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def __call__(self, *, prompt, attempt_number):
        self.calls.append(
            {
                "prompt": prompt,
                "attempt_number": attempt_number,
            }
        )
        return next(self.outputs)


class MappingEvaluator:
    def __init__(self, reports):
        self.reports = dict(reports)

    def __call__(self, output):
        return self.reports[output]


class CarePlanOrchestratorTests(TestCase):
    def test_simple_path_routes_and_completes(self):
        orchestrator = CarePlanOrchestrator()
        generator_instances = []

        def generator_factory(*, prompt_version, model):
            self.assertEqual(prompt_version, "v1")
            self.assertEqual(model, "gpt-5.5")
            generator = SequenceGenerator([GeneratedOutput("clean", 0.11)])
            generator_instances.append(generator)
            return generator

        evaluator = MappingEvaluator(
            {
                "clean": HallucinationReport(
                    valid=True,
                    hallucination_count=0,
                )
            }
        )

        result = orchestrator.process(
            patient_record={
                "diagnoses": ["G70.00"],
                "medications": ["IVIG", "Pyridostigmine"],
                "allergies": [],
                "lab_values": {"SCr": 0.8},
            },
            generator_factory=generator_factory,
            evaluator=evaluator,
            initial_prompt="base prompt",
        )

        self.assertEqual(result.strategy.complexity, "simple")
        self.assertEqual(result.prompt_version, "v1")
        self.assertEqual(result.model, "gpt-5.5")
        self.assertEqual(result.status, LoopStatus.SUCCESS)
        self.assertEqual(result.output, "clean")
        self.assertIn("router selected complexity=simple", result.decision_log[5])
        self.assertIn("loop completed status=success", result.decision_log[7])
        self.assertEqual(generator_instances[0].calls[0]["prompt"], "base prompt")

    def test_complex_path_keeps_decision_log_and_repairs(self):
        orchestrator = CarePlanOrchestrator()

        def generator_factory(*, prompt_version, model):
            self.assertEqual(prompt_version, "v2")
            self.assertEqual(model, "claude-sonnet-4-6")
            return SequenceGenerator(
                [
                    GeneratedOutput("bad", 0.10),
                    GeneratedOutput("good", 0.20),
                ]
            )

        evaluator = MappingEvaluator(
            {
                "bad": HallucinationReport(
                    valid=False,
                    hallucination_count=1,
                    findings=[
                        HallucinationFinding(
                            path="$.problems[0]",
                            message="Unsupported claim",
                        )
                    ],
                ),
                "good": HallucinationReport(
                    valid=True,
                    hallucination_count=0,
                ),
            }
        )

        result = orchestrator.process(
            patient_record={
                "diagnoses": ["G70.00", "N18.9"],
                "medications": ["IVIG", "Pyridostigmine"],
                "allergies": [],
                "lab_values": {},
            },
            generator_factory=generator_factory,
            evaluator=evaluator,
            initial_prompt="base prompt",
        )

        self.assertEqual(result.strategy.complexity, "complex")
        self.assertEqual(result.prompt_version, "v2")
        self.assertEqual(result.model, "claude-sonnet-4-6")
        self.assertEqual(result.status, LoopStatus.SUCCESS)
        self.assertEqual(result.output, "good")
        self.assertIn("router selected complexity=complex", result.decision_log[5])
        self.assertIn("loop completed status=success", result.decision_log[7])
        self.assertIn(
            "attempt=2 hallucinations 1->0 delta=-1 cost=0.2000",
            result.decision_log,
        )
