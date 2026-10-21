import uuid
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.core.exceptions import ValidationError, PermissionDenied
from django.core.management import call_command, CommandError
from django.test import TestCase, Client, override_settings
from django.utils import timezone
from care.models import Patient, Provider, Order, GenerationJob, ReviewDecision
from care.services import patient_snapshot, review_care_plan, retry_order
from care.test_structured_llm import parsed_care_plan


class ReviewTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username="reviewer", is_staff=True)
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label="care", codename="review_careplan"))
        patient = Patient.objects.create(mrn="000123", first_name="Ada", last_name="Patient")
        provider = Provider.objects.create(npi="1234567890", name="Test Provider")
        self.order = Order.objects.create(patient=patient, provider=provider, medication_name="IVIG", primary_diagnosis="G70.01")
        self.job = GenerationJob.objects.create(order=self.order, request_key=uuid.uuid4(), input_snapshot=patient_snapshot(self.order), status="needs_review", plan=parsed_care_plan().model_dump(mode="json"), completed_at=timezone.now())
        self.client.force_login(self.user)
        self.url = f"/jobs/{self.job.pk}/review/"

    def review(self, decision="approved", comments=""):
        return review_care_plan(job_id=self.job.pk, reviewer=self.user, decision=decision, comments=comments)

    def test_approval_persists_reviewer_and_labels_page_and_export(self):
        response = self.client.post(self.url, {"decision": "approved", "comments": "Reviewed all evidence."})
        self.assertEqual(response.status_code, 302)
        review = ReviewDecision.objects.get()
        self.assertEqual(review.reviewer, self.user)
        self.assertEqual(review.comments, "Reviewed all evidence.")
        self.assertEqual(len(review.plan_sha256), 64)
        self.assertIsNotNone(review.created_at)
        page = self.client.get(f"/order/{self.order.pk}/")
        self.assertContains(page, "Care Plan — Approved")
        self.assertNotContains(page, "Approve plan")
        self.assertNotContains(page, "Draft only.")
        export = self.client.get(f"/order/{self.order.pk}/download/")
        self.assertContains(export, "APPROVED")
        self.assertContains(export, "Reviewed by: reviewer")
        self.assertEqual(self.client.get(f"/api/jobs/{self.job.pk}/").json()["review_status"], "approved")

    def test_rejection_and_changes_require_comments(self):
        for decision in ("rejected", "changes_requested"):
            with self.assertRaises(ValidationError):
                self.review(decision, "   ")
        self.assertFalse(ReviewDecision.objects.exists())
        self.assertEqual(self.client.post(self.url, {"decision": "rejected"}).status_code, 409)

    def test_rejected_plan_can_regenerate_without_inheriting_approval(self):
        self.review("rejected", "Evidence does not support the plan.")
        self.assertContains(self.client.get(f"/order/{self.order.pk}/download/"), "REJECTED")
        new = retry_order(self.order, uuid.uuid4())
        self.assertEqual(new.status, "queued")
        self.assertFalse(ReviewDecision.objects.filter(job=new).exists())
        self.assertEqual(ReviewDecision.objects.get(job=self.job).decision, "rejected")

    def test_approved_plan_cannot_be_regenerated_by_retry(self):
        self.review()
        with self.assertRaises(ValidationError):
            retry_order(self.order, uuid.uuid4())

    def test_decision_is_idempotent_and_cannot_be_overwritten(self):
        first = self.review(comments="Checked")
        self.assertEqual(self.review(comments="Checked").pk, first.pk)
        with self.assertRaises(ValidationError):
            self.review("rejected", "Different decision")
        first.comments = "Edited"
        with self.assertRaises(ValueError):
            first.save()
        with self.assertRaises(ValueError):
            first.delete()
        self.assertEqual(ReviewDecision.objects.count(), 1)

    def test_staff_without_review_permission_cannot_review(self):
        staff = get_user_model().objects.create_user(username="intake", is_staff=True)
        with self.assertRaises(PermissionDenied):
            review_care_plan(job_id=self.job.pk, reviewer=staff, decision="approved")
        self.client.force_login(staff)
        self.assertEqual(self.client.post(self.url, {"decision": "approved"}).status_code, 403)
        self.assertNotContains(self.client.get(f"/order/{self.order.pk}/"), "Approve plan")

    def test_incomplete_or_stale_job_cannot_be_approved(self):
        for state in ("queued", "running", "failed"):
            GenerationJob.objects.filter(pk=self.job.pk).update(status=state)
            with self.assertRaises(ValidationError):
                self.review()
        GenerationJob.objects.filter(pk=self.job.pk).update(status="needs_review")
        GenerationJob.objects.create(order=self.order, request_key=uuid.uuid4())
        with self.assertRaises(ValidationError):
            self.review()

    def test_post_and_csrf_required(self):
        self.assertEqual(self.client.get(self.url).status_code, 405)
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.user)
        self.assertEqual(client.post(self.url, {"decision": "approved"}).status_code, 403)

    def test_unknown_decision_rejected(self):
        with self.assertRaises(ValidationError):
            self.review("auto_approved")

    def test_demo_seed_refuses_application_database(self):
        with self.assertRaises(CommandError):
            call_command("seed_approval_demo", password="test-only")

    @override_settings(DEMO_MODE=True)
    def test_demo_disables_live_submission_and_retry(self):
        self.assertEqual(self.client.post("/", {}).status_code, 405)
        self.assertEqual(self.client.post(f"/order/{self.order.pk}/retry/", {}).status_code, 409)
        self.assertContains(self.client.get("/"), "Fictional plans only")
