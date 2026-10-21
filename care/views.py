import uuid

from django.shortcuts import render, get_object_or_404, redirect
from django.http import (
    HttpResponse,
    HttpResponseNotAllowed,
    JsonResponse,
)

from .models import GenerationAudit, GenerationJob, Order

from django.contrib.admin.views.decorators import staff_member_required
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.conf import settings
from django.views.decorators.http import require_POST
from .forms import OrderForm
from .services import submit_order, retry_order, review_care_plan


STAGE_LABELS = {
    "queued": "Order saved — waiting for worker",
    "evidence_preparation": "Preparing and caching medication label embeddings",
    "routing": "Selecting generation workflow",
    "retrieval": "Retrieving drug-label evidence",
    "generation": "Generating structured care plan",
    "validation": "Validating plan structure",
    "verification": "Checking patient facts and citation IDs",
    "repair": "Repairing flagged issues",
    "needs_review": "Waiting for pharmacist review",
}


@staff_member_required(login_url="login")
def order_form(request):
    initial = {"request_key": uuid.uuid4()}
    if settings.DEMO_MODE and request.GET.get("example") == "1":
        initial.update({
            "mrn": str(100000 + uuid.uuid4().int % 900000),
            "first_name": "Alex", "last_name": "Demo", "dob": "1980-01-01",
            "provider_npi": "9999999999", "provider_name": "Fictional Provider",
            "primary_diagnosis": "E11.9", "medication_name": "metformin",
            "weight_kg": "72", "allergies": "Penicillin",
            "medication_history": "Metformin immediate-release 500 mg orally twice daily with meals.",
            "patient_records": "Fictional training case. Type 2 diabetes; HbA1c 7.8%; eGFR 85 mL/min/1.73 m2. No reported adverse effects. Confirm all details before clinical use.",
        })
    form = OrderForm(request.POST if request.method == "POST" else None, initial=initial)
    if request.method == "POST" and form.is_valid():
        try:
            order = submit_order(form.cleaned_data)
            return redirect("care_plan", pk=order.pk)
        except ValidationError as exc:
            form.add_error(None, exc)
        except IntegrityError:
            existing = GenerationJob.objects.filter(request_key=form.cleaned_data["request_key"]).first()
            if existing:
                return redirect("care_plan", pk=existing.order_id)
            form.add_error(None, "Another submission changed this record. Please check and submit again.")
    return render(request, "care/form.html", {"form": form, "demo": settings.DEMO_MODE, "recent_orders": Order.objects.select_related("patient").order_by("-created_at")[:20]})


@staff_member_required(login_url="login")
def care_plan(request, pk):
    order = get_object_or_404(Order.objects.select_related("patient", "provider"), pk=pk)
    job = order.jobs.first()
    review = getattr(job, "review", None) if job else None
    return render(request, "care/care_plan.html", {
        "order": order, "job": job, "retry_key": uuid.uuid4(),
        "stage_label": STAGE_LABELS.get(job.stage, job.stage.replace("_", " ").capitalize()) if job else "",
        "pipeline_events": [{**event, "label": STAGE_LABELS.get(event["stage"], event["stage"].replace("_", " ").capitalize())} for event in job.events] if job else [],
        "review": review, "demo": settings.DEMO_MODE,
        "can_review": request.user.has_perm("care.review_careplan"),
        "can_retry": job and (job.status == "failed" or (review and review.decision != "approved")),
        "monitoring_items": [item["monitoring"] for item in (job.plan or {}).get("interventions", []) if item.get("monitoring")] if job else [],
        "active": job is not None and job.status in ("queued", "running"),
    })


def render_plan(plan, review=None):
    heading = review.get_decision_display().upper() if review else "DRAFT — requires pharmacist review"
    lines = [heading, ""]
    if review:
        lines += [f"Reviewed by: {review.reviewer_name}", f"Reviewed at: {review.created_at.isoformat()}", f"Comments: {review.comments}", ""]
    for title, key in (("Problem List", "problems"), ("Goals", "goals"), ("Pharmacist Interventions", "interventions")):
        lines.append(title)
        for item in plan.get(key, []):
            lines.append("; ".join(f"{k}: {v}" for k, v in item.items() if v))
        lines.append("")
    return "\n".join(lines)


@staff_member_required(login_url="login")
def download_care_plan(request, pk):
    order = get_object_or_404(Order, pk=pk)
    job = order.jobs.first()
    if job:
        if job.status != "needs_review" or not job.plan:
            return HttpResponse("A draft is not ready for download.", status=409)
        text = render_plan(job.plan, getattr(job, "review", None))
        if settings.DEMO_MODE:
            text = "DEMO — NOT FOR CLINICAL USE\n\n" + text
    else:
        text = order.care_plan
    response = HttpResponse(text, content_type="text/plain; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="care_plan_{order.pk}.txt"'
    return response


@staff_member_required(login_url="login")
def job_status(request, pk):
    job = get_object_or_404(GenerationJob, pk=pk)
    review = getattr(job, "review", None)
    return JsonResponse({"review_status": review.decision if review else "pending", "job_id": str(job.pk), "order_id": job.order_id,
                         "status": job.status, "stage": job.stage, "error": job.error,
                         "events": job.events, "updated_at": job.updated_at.isoformat()})


@staff_member_required(login_url="login")
@require_POST
def retry_generation(request, pk):
    order = get_object_or_404(Order, pk=pk)
    try:
        key = uuid.UUID(request.POST.get("request_key", ""))
        retry_order(order, key)
    except (ValueError, ValidationError, IntegrityError):
        return HttpResponse("This generation cannot be retried. Refresh the order page for its current status.", status=409)
    return redirect("care_plan", pk=pk)


@staff_member_required(login_url="login")
def llm_logs(request):
    """Return generation audit records for one order."""
    if request.method != "GET":
        return HttpResponseNotAllowed(["GET"])

    order_id = request.GET.get("order_id", "").strip()
    if not order_id:
        return JsonResponse(
            {"error": "order_id query parameter is required"},
            status=400,
        )
    try:
        order_id_value = int(order_id)
    except ValueError:
        return JsonResponse(
            {"error": "order_id must be an integer"},
            status=400,
        )

    try:
        limit = min(max(int(request.GET.get("limit", "50")), 1), 100)
        offset = max(int(request.GET.get("offset", "0")), 0)
    except ValueError:
        return JsonResponse(
            {"error": "limit and offset must be integers"},
            status=400,
        )

    queryset = GenerationAudit.objects.filter(order_id=order_id_value)
    total = queryset.count()
    audits = queryset[offset : offset + limit]
    return JsonResponse(
        {
            "order_id": order_id_value,
            "count": total,
            "limit": limit,
            "offset": offset,
            "results": [_serialize_audit(audit) for audit in audits],
        }
    )


def _serialize_audit(audit):
    return {
        "run_id": str(audit.run_id),
        "run_group_id": str(audit.run_group_id),
        "attempt_number": audit.attempt_number,
        "order_id": audit.order_id,
        "workflow": audit.workflow,
        "provider": audit.provider,
        "model": audit.model,
        "model_parameters": audit.model_parameters,
        "prompt": {
            "workflow": audit.prompt_workflow,
            "requested_version": audit.prompt_requested_version,
            "version": audit.prompt_version,
            "checksum": audit.prompt_checksum,
            "rendered_sha256": audit.rendered_prompt_sha256,
            "rendered": audit.rendered_prompt,
        },
        "input_snapshot": audit.input_snapshot,
        "input_sha256": audit.input_sha256,
        "raw_output": audit.raw_output,
        "raw_response": audit.raw_response,
        "output_sha256": audit.output_sha256,
        "token_usage": audit.token_usage,
        "input_tokens": audit.input_tokens,
        "output_tokens": audit.output_tokens,
        "cache_creation_input_tokens": audit.cache_creation_input_tokens,
        "cache_read_input_tokens": audit.cache_read_input_tokens,
        "duration_ms": audit.duration_ms,
        "parse_success": audit.parse_success,
        "status": audit.status,
        "stop_reason": audit.stop_reason,
        "provider_request_id": audit.provider_request_id,
        "error_type": audit.error_type,
        "error_message": audit.error_message,
        "validation_errors": audit.validation_errors,
        "started_at": audit.started_at.isoformat(),
        "completed_at": (
            audit.completed_at.isoformat() if audit.completed_at else None
        ),
        "record_sha256": audit.record_sha256,
    }


@staff_member_required(login_url="login")
@require_POST
def submit_review(request, pk):
    job = get_object_or_404(GenerationJob, pk=pk)
    try:
        review_care_plan(job_id=job.pk, reviewer=request.user,
                         decision=request.POST.get("decision", ""),
                         comments=request.POST.get("comments", ""))
    except ValidationError as exc:
        return render(request, "care/review_error.html", {"errors": exc.messages, "order": job.order}, status=409)
    except IntegrityError:
        return render(request, "care/review_error.html", {"errors": ["This version has already been reviewed. Refresh the page."], "order": job.order}, status=409)
    return redirect("care_plan", pk=job.order_id)
