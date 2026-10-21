import uuid
from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.test import TestCase, Client, override_settings
from django.utils import timezone

from rag.retrieve import RetrievedChunk, format_reference_material
from .forms import OrderForm
from .models import GenerationAudit, GenerationJob, Order
from .services import submit_order, retry_order
from .worker import claim_next, run_job, fail_stale_jobs
from .test_structured_llm import parsed_care_plan
from .structured_llm import StructuredOutputError


class JobTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username="staff", password="test-password", is_staff=True)
        self.client.force_login(self.user)
        self.data = dict(request_key=str(uuid.uuid4()), provider_npi="1234567890", provider_name="Test Provider", mrn="000123", first_name="Ada", last_name="Patient", dob="1979-06-08", primary_diagnosis="G70.01", medication_name="IVIG", weight_kg="72", allergies="Penicillin", additional_diagnoses="", medication_history="", patient_records="")
        self.chunk = RetrievedChunk("chunk-1", "IVIG label evidence", "hash", "IVIG", "Warnings", 0.1)

    def submit(self):
        form = OrderForm(self.data)
        self.assertTrue(form.is_valid(), form.errors)
        return submit_order(form.cleaned_data)

    def run_with_plan(self, plan=None):
        with patch("care.worker.retrieve_chunks", return_value=[self.chunk]), patch("care.structured_llm.generate_structured_care_plan", return_value=plan or parsed_care_plan()) as generate:
            run_job(claim_next())
        return generate

    def test_submit_is_async_and_idempotent(self):
        with patch("care.structured_llm.generate_structured_care_plan") as generate:
            first = self.client.post("/", self.data)
            second = self.client.post("/", self.data)
        self.assertEqual(first.status_code, 302)
        self.assertEqual(first.url, second.url)
        self.assertEqual(Order.objects.count(), 1)
        self.assertEqual(GenerationJob.objects.get().status, "queued")
        generate.assert_not_called()

    def test_worker_persists_plan_evidence_report_and_audit_context(self):
        order = self.submit()
        generate = self.run_with_plan()
        job = order.jobs.get()
        self.assertEqual(job.status, "needs_review")
        self.assertEqual(job.plan["patient_summary"]["mrn"], "000123")
        self.assertTrue(job.verification["report"]["valid"])
        self.assertEqual(job.evidence[0]["chunk_id"], "chunk-1")
        self.assertTrue(generate.call_args.kwargs["audit_enabled"])
        self.assertEqual(generate.call_args.kwargs["audit_run_group_id"], job.pk)
        self.assertIn("claude", generate.call_args.kwargs["model"])
        self.assertIn("[chunk-1]", generate.call_args.kwargs["reference_material"])
        self.assertContains(self.client.get(f"/order/{order.pk}/"), "pharmacist must review")
        self.assertContains(self.client.get(f"/order/{order.pk}/download/"), "DRAFT")

    @override_settings(DEMO_MODE=True)
    def test_demo_submission_reaches_review_with_visible_pipeline(self):
        response = self.client.post("/", self.data)
        self.assertEqual(response.status_code, 302)
        self.run_with_plan()
        job = GenerationJob.objects.get()
        self.assertEqual(job.status, "needs_review")
        self.assertFalse(hasattr(job, "review"))
        page = self.client.get(response.url)
        self.assertContains(page, "Waiting for pharmacist review.")
        self.assertContains(page, "Retrieving drug-label evidence")
        self.assertContains(page, "Generating structured care plan")
        self.assertContains(page, "Checking patient facts and citation IDs")

    def test_real_adapter_records_job_group(self):
        from types import SimpleNamespace
        order = self.submit()
        response = SimpleNamespace(id="test-response", stop_reason="end_turn", parsed_output=parsed_care_plan(), content=[], usage={})
        with patch("care.worker.retrieve_chunks", return_value=[self.chunk]), patch("care.structured_llm.anthropic.Anthropic") as client:
            client.return_value.messages.parse.return_value = response
            run_job(claim_next())
        audit = GenerationAudit.objects.get()
        self.assertEqual(audit.run_group_id, order.jobs.get().pk)
        self.assertEqual(audit.order_id, order.pk)

    def test_missing_evidence_fails_without_generation(self):
        order = self.submit()
        with patch("care.worker.retrieve_chunks", return_value=[]), patch("care.structured_llm.generate_structured_care_plan") as generate:
            run_job(claim_next())
        self.assertEqual(order.jobs.get().status, "failed")
        generate.assert_not_called()

    def test_invalid_citations_exhaust_bounded_repairs(self):
        order = self.submit()
        plan = parsed_care_plan()
        plan.goals[0].source_ids = ["invented"]
        generate = self.run_with_plan(plan)
        job = order.jobs.get()
        self.assertEqual(job.status, "failed")
        self.assertEqual(generate.call_count, 2)
        self.assertFalse(job.verification["report"]["valid"])
        self.assertIsNotNone(job.plan)

    def test_patient_mismatch_requires_review_without_auto_accept(self):
        order = self.submit()
        plan = parsed_care_plan()
        plan.patient_summary.weight_kg = 99
        self.run_with_plan(plan)
        job = order.jobs.get()
        self.assertEqual(job.status, "needs_review")
        self.assertTrue(job.verification["report"]["escalate_to_pharmacist"])

    def test_parse_exhaustion_is_terminal(self):
        order = self.submit()
        with patch("care.worker.retrieve_chunks", return_value=[self.chunk]), patch("care.structured_llm.generate_structured_care_plan", side_effect=StructuredOutputError("bad output")) as generate:
            run_job(claim_next())
        self.assertEqual(order.jobs.get().status, "failed")
        self.assertEqual(generate.call_count, 3)

    def test_provider_exception_is_safe_and_retry_preserves_history(self):
        order = self.submit()
        with patch("care.worker.retrieve_chunks", side_effect=RuntimeError("secret patient content")):
            run_job(claim_next())
        previous = order.jobs.get()
        self.assertNotIn("secret", previous.error)
        key = uuid.uuid4()
        new = retry_order(order, key)
        self.assertEqual(retry_order(order, key).pk, new.pk)
        self.assertEqual(order.jobs.count(), 2)
        self.assertEqual(new.input_snapshot, previous.input_snapshot)
        with self.assertRaises(ValidationError):
            retry_order(order, uuid.uuid4())

    def test_claim_only_once_and_database_prevents_two_active_jobs(self):
        order = self.submit()
        self.assertIsNotNone(claim_next())
        self.assertIsNone(claim_next())
        with self.assertRaises(IntegrityError), transaction.atomic():
            GenerationJob.objects.create(order=order, request_key=uuid.uuid4())

    def test_stale_recovery_prevents_late_worker_overwrite(self):
        order = self.submit()
        job = claim_next()
        GenerationJob.objects.filter(pk=job.pk).update(updated_at=timezone.now() - timedelta(hours=1))
        self.assertEqual(fail_stale_jobs(), 1)
        with patch("care.worker.retrieve_chunks") as retrieve:
            run_job(job)
        retrieve.assert_not_called()
        self.assertEqual(order.jobs.get().status, "failed")

    def test_input_snapshot_does_not_follow_later_order_edits(self):
        order = self.submit()
        order.medication_name = "Changed"
        order.save()
        self.assertEqual(order.jobs.get().input_snapshot["medication_name"], "IVIG")

    def test_conflicting_patient_rolls_back(self):
        self.submit()
        self.data.update(request_key=str(uuid.uuid4()), first_name="Different")
        response = self.client.post("/", self.data)
        self.assertContains(response, "different patient details")
        self.assertEqual(Order.objects.count(), 1)

    def test_invalid_form_returns_field_errors(self):
        self.data["weight_kg"] = "not a number"
        self.assertContains(self.client.post("/", self.data), "Enter a number")
        self.assertFalse(Order.objects.exists())

    def test_staff_auth_and_csrf_required(self):
        order = self.submit()
        self.client.logout()
        for url in ("/", f"/order/{order.pk}/", f"/order/{order.pk}/download/", f"/api/jobs/{order.jobs.get().pk}/", "/api/llm-logs?order_id=1"):
            self.assertEqual(self.client.get(url).status_code, 302)
        ordinary = get_user_model().objects.create_user(username="ordinary")
        self.client.force_login(ordinary)
        self.assertEqual(self.client.get(f"/order/{order.pk}/").status_code, 302)
        csrf_client = Client(enforce_csrf_checks=True)
        csrf_client.force_login(self.user)
        self.assertEqual(csrf_client.post("/", self.data).status_code, 403)

    def test_polling_and_retry_http(self):
        order = self.submit()
        job = order.jobs.get()
        response = self.client.get(f"/api/jobs/{job.pk}/")
        self.assertEqual(response.json()["status"], "queued")
        self.assertEqual(self.client.get(f"/order/{order.pk}/download/").status_code, 409)
        self.assertEqual(self.client.get(f"/order/{order.pk}/retry/").status_code, 405)
        GenerationJob.objects.filter(pk=job.pk).update(status="failed")
        self.assertEqual(self.client.post(f"/order/{order.pk}/retry/", {"request_key": str(uuid.uuid4())}).status_code, 302)

    def test_reference_format_preserves_ids(self):
        self.assertIn("[chunk-1] IVIG", format_reference_material([self.chunk]))

    def test_graph_reverifies_deterministic_repairs(self):
        from .care_plan_graph import build_care_plan_graph, make_initial_state
        from .generation_loop import HallucinationFinding, HallucinationReport
        import json
        seen = []
        def evaluate(output):
            weight = json.loads(output)["patient_summary"]["weight_kg"]
            seen.append(weight)
            findings = [] if weight == 72 else [HallucinationFinding("$.patient_summary.weight_kg", "Wrong weight", risk_level="low")]
            return HallucinationReport(not findings, len(findings), findings)
        plan = parsed_care_plan()
        plan.patient_summary.weight_kg = 99
        graph = build_care_plan_graph(evaluator=evaluate)
        with patch("care.structured_llm.generate_structured_care_plan", return_value=plan):
            final = graph.invoke(make_initial_state(order_id=1, patient_data={"weight_kg": 72}, initial_prompt="evidence"))
        self.assertEqual(seen, [99, 72])
        self.assertEqual(final["status"], "success")

    def test_graph_cannot_silently_skip_verification(self):
        from .care_plan_graph import build_care_plan_graph, make_initial_state
        with patch("care.structured_llm.generate_structured_care_plan", return_value=parsed_care_plan()):
            with self.assertRaisesRegex(RuntimeError, "evaluator is required"):
                build_care_plan_graph().invoke(make_initial_state(order_id=1, patient_data={}, initial_prompt="evidence"))
