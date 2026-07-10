import uuid
from datetime import date
from unittest.mock import patch

from django.test import TestCase

from care.audit import AuditRecorder, sha256_text, snapshot_order
from care.models import GenerationAudit, Order, Patient, Provider


class AuditRecorderTests(TestCase):
    def setUp(self):
        provider = Provider.objects.create(
            npi="1234567890",
            name="Dr. Test",
        )
        patient = Patient.objects.create(
            mrn="000123",
            first_name="Ada",
            last_name="Patient",
            dob=date(1979, 6, 8),
        )
        self.order = Order.objects.create(
            patient=patient,
            provider=provider,
            primary_diagnosis="G70.01",
            medication_name="IVIG",
            weight_kg="72",
            allergies="Penicillin",
            patient_records="Baseline SCr 0.8 mg/dL",
        )

    def start_recorder(self, **overrides):
        values = {
            "workflow": "test_generation",
            "provider": "anthropic",
            "model": "test-model",
            "rendered_prompt": "Patient: Ada",
            "input_snapshot": snapshot_order(self.order),
            "model_parameters": {
                "max_tokens": 2048,
                "temperature": 0,
            },
            "prompt_metadata": {
                "prompt_workflow": "careplan_generation",
                "prompt_requested_version": "current",
                "prompt_version": "v3",
                "prompt_checksum": "abc123",
            },
            "order": self.order,
        }
        values.update(overrides)
        return AuditRecorder.start(**values)

    def test_success_records_complete_run_spec(self):
        recorder = self.start_recorder()

        audit = recorder.succeed(
            raw_output='{"problems": []}',
            raw_response={"id": "msg_123", "content": [{"type": "text"}]},
            token_usage={
                "input_tokens": 120,
                "output_tokens": 45,
                "cache_read_input_tokens": 20,
            },
            provider_request_id="msg_123",
            stop_reason="end_turn",
            parse_success=True,
        )

        audit.refresh_from_db()
        self.assertEqual(audit.status, GenerationAudit.Status.SUCCEEDED)
        self.assertEqual(audit.prompt_version, "v3")
        self.assertEqual(audit.model_parameters["max_tokens"], 2048)
        self.assertEqual(audit.input_snapshot["patient"]["mrn"], "000123")
        self.assertEqual(audit.input_tokens, 120)
        self.assertEqual(audit.output_tokens, 45)
        self.assertEqual(audit.provider_request_id, "msg_123")
        self.assertIs(audit.parse_success, True)
        self.assertEqual(
            audit.output_sha256,
            sha256_text('{"problems": []}'),
        )
        self.assertTrue(audit.record_sha256)
        self.assertIsNotNone(audit.completed_at)

    def test_failure_records_error(self):
        recorder = self.start_recorder()

        audit = recorder.fail(RuntimeError("provider unavailable"))

        self.assertEqual(audit.status, GenerationAudit.Status.FAILED)
        self.assertEqual(audit.error_type, "RuntimeError")
        self.assertEqual(audit.error_message, "provider unavailable")

    def test_finalized_record_is_immutable(self):
        audit = self.start_recorder().succeed(
            raw_output="output",
            raw_response={},
        )
        audit.status = GenerationAudit.Status.FAILED

        with self.assertRaisesRegex(ValueError, "immutable"):
            audit.save()

        with self.assertRaisesRegex(ValueError, "cannot be deleted"):
            audit.delete()

    def test_retry_attempts_share_group_and_are_unique(self):
        group_id = uuid.uuid4()
        first = self.start_recorder(
            run_group_id=group_id,
            attempt_number=1,
        ).fail(ValueError("invalid"))
        second = self.start_recorder(
            run_group_id=group_id,
            attempt_number=2,
        ).succeed(raw_output="valid", raw_response={})

        self.assertEqual(first.run_group_id, second.run_group_id)
        self.assertEqual(
            GenerationAudit.objects.filter(run_group_id=group_id).count(),
            2,
        )

    @patch("care.llm.anthropic.Anthropic")
    def test_live_generation_creates_audit(self, anthropic_client):
        from types import SimpleNamespace

        from care.llm import generate_care_plan

        message = SimpleNamespace(
            id="msg_live",
            content=[
                SimpleNamespace(type="text", text="Generated care plan")
            ],
            model="claude-sonnet-4-6",
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=25,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                model_dump=lambda mode: {
                    "input_tokens": 100,
                    "output_tokens": 25,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            ),
            model_dump=lambda mode: {
                "id": "msg_live",
                "content": [
                    {"type": "text", "text": "Generated care plan"}
                ],
                "stop_reason": "end_turn",
            },
        )
        anthropic_client.return_value.messages.create.return_value = message

        output = generate_care_plan(self.order)

        self.assertEqual(output, "Generated care plan")
        audit = GenerationAudit.objects.get(order=self.order)
        self.assertEqual(audit.status, GenerationAudit.Status.SUCCEEDED)
        self.assertEqual(audit.raw_output, output)
        self.assertEqual(audit.input_tokens, 100)

    def test_llm_logs_api_filters_by_order(self):
        self.start_recorder().succeed(
            raw_output="complete output",
            raw_response={"id": "msg_api"},
            token_usage={"input_tokens": 10, "output_tokens": 5},
            provider_request_id="msg_api",
            parse_success=True,
        )

        response = self.client.get(
            "/api/llm-logs",
            {"order_id": self.order.pk},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["order_id"], self.order.pk)
        self.assertEqual(payload["count"], 1)
        record = payload["results"][0]
        self.assertEqual(record["order_id"], self.order.pk)
        self.assertEqual(record["prompt"]["version"], "v3")
        self.assertEqual(record["model"], "test-model")
        self.assertEqual(record["input_tokens"], 10)
        self.assertEqual(record["output_tokens"], 5)
        self.assertEqual(record["raw_output"], "complete output")
        self.assertIs(record["parse_success"], True)
        self.assertIsInstance(record["duration_ms"], int)

    def test_llm_logs_api_requires_order_id(self):
        response = self.client.get("/api/llm-logs")

        self.assertEqual(response.status_code, 400)
        self.assertIn("order_id", response.json()["error"])

    def test_structured_generation_records_parse_success(self):
        from types import SimpleNamespace

        from care.structured_llm import generate_anthropic_care_plan
        from care.test_structured_llm import parsed_care_plan

        response = SimpleNamespace(
            id="msg_structured",
            stop_reason="end_turn",
            parsed_output=parsed_care_plan(),
            content=[
                SimpleNamespace(
                    type="text",
                    text=parsed_care_plan().model_dump_json(),
                )
            ],
            usage={
                "input_tokens": 200,
                "output_tokens": 80,
            },
        )
        client = SimpleNamespace(
            messages=SimpleNamespace(parse=lambda **kwargs: response)
        )

        generate_anthropic_care_plan(
            patient_record="Patient record",
            reference_material="[chunk-1] Evidence",
            client=client,
            audit_enabled=True,
            audit_order=self.order,
        )

        audit = GenerationAudit.objects.get(
            provider_request_id="msg_structured"
        )
        self.assertIs(audit.parse_success, True)
        self.assertEqual(audit.input_tokens, 200)
        self.assertEqual(audit.output_tokens, 80)
        self.assertEqual(audit.prompt_version, "v2")
