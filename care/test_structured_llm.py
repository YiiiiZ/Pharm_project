from types import SimpleNamespace
from unittest import TestCase

from care.care_plan_schema import CarePlan
from care.structured_llm import (
    StructuredOutputError,
    generate_anthropic_care_plan,
    generate_openai_care_plan,
)


def parsed_care_plan() -> CarePlan:
    return CarePlan.model_validate(
        {
            "patient_summary": {
                "patient_name": "Ada Patient",
                "mrn": "000123",
                "date_of_birth": "1979-06-08",
                "weight_kg": 72,
                "allergies": ["Penicillin"],
                "medication_name": "IVIG",
                "primary_diagnosis": "G70.01",
            },
            "problems": [
                {
                    "title": "Renal injury risk",
                    "description": "IVIG may cause renal dysfunction.",
                    "source_ids": ["chunk-1"],
                }
            ],
            "goals": [
                {
                    "description": "Preserve renal function",
                    "target": "No clinically significant SCr increase",
                    "source_ids": ["chunk-1"],
                }
            ],
            "interventions": [
                {
                    "action": "Review baseline SCr and BUN.",
                    "rationale": "Assess renal risk before infusion.",
                    "source_ids": ["chunk-1"],
                }
            ],
        }
    )


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class StructuredLLMTests(TestCase):
    def test_openai_uses_pydantic_schema_and_returns_model(self):
        responses = FakeResponses(
            SimpleNamespace(status="completed", output_parsed=parsed_care_plan())
        )
        client = SimpleNamespace(responses=responses)

        result = generate_openai_care_plan(
            patient_record="Patient record",
            reference_material="[chunk-1] Label evidence",
            client=client,
            model="test-openai",
        )

        self.assertIsInstance(result, CarePlan)
        self.assertIs(responses.kwargs["text_format"], CarePlan)
        self.assertEqual(responses.kwargs["model"], "test-openai")

    def test_openai_appends_repair_instructions(self):
        responses = FakeResponses(
            SimpleNamespace(status="completed", output_parsed=parsed_care_plan())
        )
        client = SimpleNamespace(responses=responses)

        generate_openai_care_plan(
            patient_record="Patient record",
            reference_material="Label evidence",
            repair_instructions="Fix $.goals[0].target",
            client=client,
        )

        content = responses.kwargs["input"][0]["content"]
        self.assertIn("Fix $.goals[0].target", content)

    def test_openai_incomplete_response_raises(self):
        client = SimpleNamespace(
            responses=FakeResponses(
                SimpleNamespace(
                    status="incomplete",
                    incomplete_details={"reason": "max_output_tokens"},
                    output_parsed=None,
                )
            )
        )

        with self.assertRaisesRegex(StructuredOutputError, "incomplete"):
            generate_openai_care_plan(
                patient_record="Patient record",
                reference_material="Reference",
                client=client,
            )

    def test_anthropic_uses_pydantic_schema_and_returns_model(self):
        messages = FakeMessages(
            SimpleNamespace(
                stop_reason="end_turn",
                parsed_output=parsed_care_plan(),
            )
        )
        client = SimpleNamespace(messages=messages)

        result = generate_anthropic_care_plan(
            patient_record="Patient record",
            reference_material="[chunk-1] Label evidence",
            client=client,
            model="test-claude",
        )

        self.assertIsInstance(result, CarePlan)
        self.assertIs(messages.kwargs["output_format"], CarePlan)
        self.assertEqual(messages.kwargs["model"], "test-claude")

    def test_anthropic_truncation_raises(self):
        client = SimpleNamespace(
            messages=FakeMessages(
                SimpleNamespace(
                    stop_reason="max_tokens",
                    parsed_output=None,
                )
            )
        )

        with self.assertRaisesRegex(StructuredOutputError, "truncated"):
            generate_anthropic_care_plan(
                patient_record="Patient record",
                reference_material="Reference",
                client=client,
            )
