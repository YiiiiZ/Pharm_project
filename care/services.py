"""Application operations shared by Django and the background worker.

The database is the queue: committing the order also commits its job. No
broker or delivery callback can lose a job between those two operations.
"""
from django.core.exceptions import ValidationError
from django.db import transaction

from .models import GenerationJob, Order, Patient, Provider


def patient_snapshot(order):
    return {
        "patient_name": f"{order.patient.first_name} {order.patient.last_name}",
        "mrn": order.patient.mrn,
        "date_of_birth": order.patient.dob.isoformat() if order.patient.dob else None,
        "weight_kg": float(order.weight_kg) if order.weight_kg else None,
        "allergies": [v.strip() for v in order.allergies.splitlines() if v.strip()],
        "medication_name": order.medication_name,
        "primary_diagnosis": order.primary_diagnosis,
        "diagnoses": [order.primary_diagnosis] + order.additional_diagnoses.splitlines(),
        "medications": [order.medication_name] + order.medication_history.splitlines(),
        "patient_records": order.patient_records,
    }


@transaction.atomic
def submit_order(data):
    existing = GenerationJob.objects.filter(request_key=data["request_key"]).first()
    if existing:
        return existing.order
    provider, _ = Provider.objects.get_or_create(
        npi=data["provider_npi"], defaults={"name": data["provider_name"]},
    )
    if provider.name.casefold() != data["provider_name"].casefold():
        raise ValidationError("This NPI belongs to a different provider name. Correct the order before submitting.")
    patient, _ = Patient.objects.get_or_create(
        mrn=data["mrn"], defaults={k: data[k] for k in ("first_name", "last_name", "dob")},
    )
    if any(getattr(patient, key) != data[key] for key in ("first_name", "last_name", "dob")):
        raise ValidationError("This MRN has different patient details. Correct the order before submitting.")
    fields = ("primary_diagnosis", "medication_name", "additional_diagnoses", "allergies", "medication_history", "patient_records")
    order = Order.objects.create(
        patient=patient, provider=provider,
        weight_kg=str(data["weight_kg"]) if data["weight_kg"] is not None else "",
        **{key: data[key] for key in fields},
    )
    GenerationJob.objects.create(order=order, request_key=data["request_key"], input_snapshot=patient_snapshot(order))
    return order


@transaction.atomic
def retry_order(order, request_key):
    """Retry a failed run without overwriting its result or input history."""
    existing = GenerationJob.objects.filter(request_key=request_key, order=order).first()
    if existing:
        return existing
    # The partial unique constraint also enforces this on SQLite.
    Order.objects.select_for_update().get(pk=order.pk)
    latest = order.jobs.first()
    review = getattr(latest, "review", None) if latest else None
    if not latest or not (latest.status == GenerationJob.Status.FAILED or (review and review.decision in ("rejected", "changes_requested"))):
        raise ValidationError("Only a failed, rejected, or changes-requested generation can be retried.")
    return GenerationJob.objects.create(
        order=order, request_key=request_key, input_snapshot=latest.input_snapshot,
    )


@transaction.atomic
def review_care_plan(*, job_id, reviewer, decision, comments=""):
    """Record one final decision against the exact generated version."""
    from django.core.exceptions import PermissionDenied
    from .audit import canonical_json, sha256_text
    from .models import ReviewDecision

    if not reviewer.is_active or not reviewer.is_staff or not reviewer.has_perm("care.review_careplan"):
        raise PermissionDenied("Pharmacist review permission is required.")
    if decision not in ReviewDecision.Decision.values:
        raise ValidationError("Choose approve, reject, or request changes.")
    comments = comments.strip()
    if len(comments) > 5000:
        raise ValidationError("Comments must be at most 5,000 characters.")
    if decision != "approved" and not comments:
        raise ValidationError("Explain why the plan is rejected or needs changes.")
    order_id = GenerationJob.objects.only("order_id").get(pk=job_id).order_id
    Order.objects.select_for_update().get(pk=order_id)
    job = GenerationJob.objects.select_for_update().get(pk=job_id)
    if job.status != "needs_review" or not job.plan or job.completed_at is None:
        raise ValidationError("Only a completed draft can be reviewed.")
    if job.order.jobs.first().pk != job.pk:
        raise ValidationError("A newer generation exists. Review the latest draft.")
    existing = ReviewDecision.objects.filter(job=job).first()
    if existing:
        if existing.reviewer_id == reviewer.pk and existing.decision == decision and existing.comments == comments:
            return existing
        raise ValidationError("This version has already been reviewed. Refresh the page.")
    return ReviewDecision.objects.create(
        job=job, reviewer=reviewer,
        reviewer_name=reviewer.get_full_name() or reviewer.get_username(),
        decision=decision, comments=comments,
        plan_sha256=sha256_text(canonical_json(job.plan)),
    )
