from unittest import TestCase

from care.generation_loop import (
    GeneratedOutput,
    HallucinationFinding,
    HallucinationReport,
    LoopStatus,
    generate_with_retry_loop,
)


class SequenceGenerator:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def __call__(self, *, prompt, attempt_number):
        self.prompts.append(
            {
                "attempt_number": attempt_number,
                "prompt": prompt,
            }
        )
        return next(self.outputs)


class MappingEvaluator:
    def __init__(self, reports):
        self.reports = dict(reports)

    def __call__(self, output):
        return self.reports[output]


class GenerateWithRetryLoopTests(TestCase):
    def test_success_on_first_attempt_records_cost_and_log(self):
        generator = SequenceGenerator([GeneratedOutput("clean", 0.12)])
        evaluator = MappingEvaluator(
            {
                "clean": HallucinationReport(
                    valid=True,
                    hallucination_count=0,
                )
            }
        )

        result = generate_with_retry_loop(
            generator,
            evaluator,
            initial_prompt="base prompt",
        )

        self.assertEqual(result.status, LoopStatus.SUCCESS)
        self.assertEqual(result.attempts_used, 1)
        self.assertEqual(result.total_cost, 0.12)
        self.assertEqual(result.original_output, "clean")
        self.assertEqual(result.final_output, "clean")
        self.assertEqual(
            result.decision_log[0],
            "attempt=1 hallucinations 0->0 delta=0 cost=0.1200",
        )
        self.assertEqual(generator.prompts[0]["prompt"], "base prompt")

    def test_failed_round_adds_feedback_into_next_prompt(self):
        generator = SequenceGenerator(
            [
                GeneratedOutput("bad", 0.10),
                GeneratedOutput("good", 0.20),
            ]
        )
        evaluator = MappingEvaluator(
            {
                "bad": HallucinationReport(
                    valid=False,
                    hallucination_count=2,
                    findings=[
                        HallucinationFinding(
                            path="$.problems[0]",
                            message="Unsupported claim",
                            evidence="No supporting source",
                        )
                    ],
                ),
                "good": HallucinationReport(
                    valid=True,
                    hallucination_count=0,
                ),
            }
        )

        result = generate_with_retry_loop(
            generator,
            evaluator,
            initial_prompt="base prompt",
            max_retries=2,
        )

        self.assertEqual(result.status, LoopStatus.SUCCESS)
        self.assertEqual(result.attempts_used, 2)
        self.assertEqual(result.retries_used, 1)
        self.assertEqual(result.final_output, "good")
        self.assertIn("Hallucination count: 2", generator.prompts[1]["prompt"])
        self.assertIn("$.problems[0]", generator.prompts[1]["prompt"])
        self.assertIn("Unsupported claim", generator.prompts[1]["prompt"])
        self.assertEqual(
            result.decision_log[1],
            "attempt=2 hallucinations 2->0 delta=-2 cost=0.2000",
        )

    def test_exhausted_retries_preserves_last_output(self):
        generator = SequenceGenerator(
            [
                GeneratedOutput("bad-1", 0.05),
                GeneratedOutput("bad-2", 0.06),
                GeneratedOutput("bad-3", 0.07),
            ]
        )
        evaluator = MappingEvaluator(
            {
                "bad-1": HallucinationReport(valid=False, hallucination_count=3),
                "bad-2": HallucinationReport(valid=False, hallucination_count=2),
                "bad-3": HallucinationReport(
                    valid=False,
                    hallucination_count=1,
                ),
            }
        )

        result = generate_with_retry_loop(
            generator,
            evaluator,
            initial_prompt="base prompt",
            max_retries=2,
        )

        self.assertEqual(result.status, LoopStatus.FAILED)
        self.assertEqual(result.attempts_used, 3)
        self.assertEqual(result.final_output, "bad-3")
        self.assertEqual(len(result.attempts), 3)
        self.assertEqual(result.total_cost, 0.18)
