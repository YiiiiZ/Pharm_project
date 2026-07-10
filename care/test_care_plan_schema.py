import json
from unittest import TestCase

from care.care_plan_schema import (
    CarePlan,
    care_plan_json_schema,
    validate_care_plan,
)


def valid_payload() -> dict:
    return {
        "patient_summary": {
            "patient_name": "Ada Patient",
            "mrn": "000123",
            "date_of_birth": "1979-06-08",
            "weight_kg": "72",
            "allergies": "Penicillin",
            "medication_name": "IVIG",
            "primary_diagnosis": "G70.01",
        },
        "problems": [
            {
                "title": "Renal injury risk",
                "description": "IVIG may cause renal dysfunction.",
                "evidence": ["Baseline renal status requires review."],
                "source_ids": ["ivig_warning_03"],
            }
        ],
        "goals": [
            {
                "description": "Preserve renal function",
                "target": "No clinically significant increase in SCr",
                "timeframe": "During treatment",
                "source_ids": ["ivig_warning_03"],
            }
        ],
        "interventions": [
            {
                "action": "Obtain baseline SCr and BUN.",
                "rationale": "Identify renal impairment before infusion.",
                "priority": "high",
                "requires_prescriber_confirmation": "yes",
                "source_ids": ["ivig_monitoring_02"],
            }
        ],
    }


class CarePlanSchemaTests(TestCase):
    def test_coerces_supported_types(self):
        result = validate_care_plan(valid_payload())

        self.assertTrue(result.valid)
        self.assertEqual(result.data["patient_summary"]["weight_kg"], 72.0)
        self.assertEqual(
            result.data["patient_summary"]["allergies"],
            ["Penicillin"],
        )
        self.assertIs(
            result.data["interventions"][0][
                "requires_prescriber_confirmation"
            ],
            True,
        )

    def test_missing_required_field_reports_exact_path(self):
        payload = valid_payload()
        del payload["goals"][0]["target"]

        result = validate_care_plan(payload)

        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].path, "$.goals[0].target")
        self.assertEqual(result.errors[0].error_type, "missing")

    def test_wrong_type_reports_exact_path(self):
        payload = valid_payload()
        payload["patient_summary"]["weight_kg"] = "seventy-two"

        result = validate_care_plan(payload)

        self.assertFalse(result.valid)
        self.assertEqual(
            result.errors[0].path,
            "$.patient_summary.weight_kg",
        )
        self.assertEqual(result.errors[0].error_type, "float_parsing")

    def test_unknown_field_is_rejected(self):
        payload = valid_payload()
        payload["unapproved_section"] = "model drift"

        result = validate_care_plan(payload)

        self.assertFalse(result.valid)
        self.assertEqual(
            result.errors[0].path,
            "$.unapproved_section",
        )
        self.assertEqual(result.errors[0].error_type, "extra_forbidden")

    def test_accepts_json_inside_markdown_fence(self):
        payload = f"```json\n{json.dumps(valid_payload())}\n```"

        result = validate_care_plan(payload)

        self.assertTrue(result.valid)

    def test_invalid_json_reports_root_error(self):
        result = validate_care_plan('{"patient_summary":')

        self.assertFalse(result.valid)
        self.assertEqual(result.errors[0].path, "$")
        self.assertEqual(result.errors[0].error_type, "json_invalid")

    def test_schema_contains_required_top_level_sections(self):
        schema = care_plan_json_schema()

        self.assertEqual(
            set(schema["required"]),
            {"patient_summary", "problems", "goals", "interventions"},
        )
        self.assertIs(CarePlan, CarePlan)
