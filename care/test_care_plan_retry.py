from unittest import TestCase

from care.care_plan_retry import (
    RetryStatus,
    build_repair_instructions,
    generate_with_validation_retry,
)
from care.test_care_plan_schema import valid_payload


class SequenceGenerator:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def __call__(self, *, attempt_number, repair_instructions):
        self.calls.append(
            {
                "attempt_number": attempt_number,
                "repair_instructions": repair_instructions,
            }
        )
        return next(self.outputs)


class CarePlanRetryTests(TestCase):
    def test_valid_first_attempt_does_not_retry(self):
        generator = SequenceGenerator([valid_payload()])

        result = generate_with_validation_retry(generator)

        self.assertEqual(result.status, RetryStatus.SUCCESS)
        self.assertEqual(result.attempts_used, 1)
        self.assertEqual(result.retries_used, 0)
        self.assertEqual(len(generator.calls), 1)

    def test_validation_error_is_sent_to_next_attempt(self):
        invalid = valid_payload()
        del invalid["goals"][0]["target"]
        generator = SequenceGenerator([invalid, valid_payload()])

        result = generate_with_validation_retry(generator)

        self.assertEqual(result.status, RetryStatus.SUCCESS)
        self.assertEqual(result.attempts_used, 2)
        feedback = generator.calls[1]["repair_instructions"]
        self.assertIn("$.goals[0].target", feedback)
        self.assertIn("Field required", feedback)
        self.assertIn('"description": "Preserve renal function"', feedback)

    def test_two_retries_then_parse_failed_preserves_outputs(self):
        outputs = [
            '{"patient_summary":',
            '{"problems": []}',
            {"unexpected": "still invalid"},
        ]
        generator = SequenceGenerator(outputs)

        result = generate_with_validation_retry(
            generator,
            max_retries=2,
        )

        self.assertEqual(result.status, RetryStatus.PARSE_FAILED)
        self.assertEqual(result.attempts_used, 3)
        self.assertEqual(result.retries_used, 2)
        self.assertEqual(result.original_output, outputs[0])
        self.assertEqual(result.final_output, outputs[-1])
        self.assertEqual(
            [attempt.raw_output for attempt in result.attempts],
            outputs,
        )
        self.assertTrue(all(not attempt.valid for attempt in result.attempts))

    def test_zero_retries_uses_one_attempt(self):
        generator = SequenceGenerator([{}])

        result = generate_with_validation_retry(
            generator,
            max_retries=0,
        )

        self.assertEqual(result.status, RetryStatus.PARSE_FAILED)
        self.assertEqual(result.attempts_used, 1)

    def test_repair_prompt_requests_complete_replacement(self):
        invalid = valid_payload()
        del invalid["patient_summary"]["mrn"]
        from care.care_plan_schema import validate_care_plan

        validation = validate_care_plan(invalid)
        prompt = build_repair_instructions(
            raw_output=invalid,
            errors=validation.errors,
        )

        self.assertIn("$.patient_summary.mrn", prompt)
        self.assertIn("complete replacement", prompt)
