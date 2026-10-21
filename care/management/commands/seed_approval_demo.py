"""Create fresh fictional review examples in the isolated demo database."""
import uuid
from datetime import date

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from care.models import GenerationJob, Order, Patient, Provider
from care.services import patient_snapshot


class Command(BaseCommand):
    help = "Add three fictional drafts to the isolated DJANGO_DEMO=true database."

    def add_arguments(self, parser):
        parser.add_argument("--password", required=True, help="Password for the demo-only pharmacist account")

    def handle(self, *args, **options):
        if not settings.DEMO_MODE:
            raise CommandError("Set DJANGO_DEMO=true. This command must not run against application records.")
        user, _ = get_user_model().objects.get_or_create(username="pharmacist")
        user.is_staff = True
        user.first_name, user.last_name = "Demo", "Pharmacist"
        user.set_password(options["password"])
        user.save()
        user.user_permissions.add(Permission.objects.get(content_type__app_label="care", codename="review_careplan"))
        patient, _ = Patient.objects.get_or_create(mrn="990001", defaults=dict(first_name="Sample", last_name="Patient", dob=date(1980, 1, 1)))
        provider, _ = Provider.objects.get_or_create(npi="9999999999", defaults={"name": "Fictional Provider"})
        for _ in range(3):
            order = Order.objects.create(patient=patient, provider=provider, medication_name="Demo medication", primary_diagnosis="Z00.00")
            snapshot = patient_snapshot(order)
            plan = {
                "patient_summary": {key: snapshot[key] for key in ("patient_name", "mrn", "date_of_birth", "weight_kg", "allergies", "medication_name", "primary_diagnosis")},
                "problems": [{"title": "Sample review item", "description": "Fictional content for testing the approval workflow.", "source_ids": ["demo-source"]}],
                "goals": [{"description": "Demonstrate a pharmacist review", "target": "Record a decision", "source_ids": ["demo-source"]}],
                "interventions": [{"action": "Inspect this fictional draft", "rationale": "Demonstrate the review interface", "priority": "low", "monitoring": "No clinical monitoring instructions in this demo.", "requires_prescriber_confirmation": False, "source_ids": ["demo-source"]}],
            }
            GenerationJob.objects.create(
                order=order, request_key=uuid.uuid4(), input_snapshot=snapshot,
                status="needs_review", stage="needs_review", plan=plan, completed_at=timezone.now(),
                evidence=[{"chunk_id": "demo-source", "product_name": "Fictional reference", "section_title": "Demo only", "text": "This is synthetic interface-test content, not a drug label or clinical guidance."}],
                verification={"scope": "Synthetic demo report. No clinical verification performed.", "report": None},
            )
        self.stdout.write(self.style.SUCCESS("Added three sample plans. Demo username: pharmacist"))
