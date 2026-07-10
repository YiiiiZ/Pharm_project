from django.shortcuts import render, get_object_or_404, redirect
from django.http import (
    HttpResponse,
    HttpResponseNotAllowed,
    JsonResponse,
)

from .models import GenerationAudit, Provider, Patient, Order
from .llm import generate_care_plan


def order_form(request):
    if request.method == "POST":
        # Pull form data
        npi = request.POST["provider_npi"].strip()
        provider_name = request.POST["provider_name"].strip()
        mrn = request.POST["mrn"].strip()
        first_name = request.POST["first_name"].strip()
        last_name = request.POST["last_name"].strip()
        dob = request.POST.get("dob", "").strip() or None
        primary_diagnosis = request.POST["primary_diagnosis"].strip()
        medication_name = request.POST["medication_name"].strip()
        additional_diagnoses = request.POST.get("additional_diagnoses", "").strip()
        weight_kg = request.POST.get("weight_kg", "").strip()
        allergies = request.POST.get("allergies", "").strip()
        medication_history = request.POST.get("medication_history", "").strip()
        patient_records = request.POST.get("patient_records", "").strip()

        # Get or create provider (NPI is the key)
        provider, _ = Provider.objects.get_or_create(
            npi=npi,
            defaults={"name": provider_name},
        )

        # Get or create patient (MRN is the key)
        patient, _ = Patient.objects.get_or_create(
            mrn=mrn,
            defaults={"first_name": first_name, "last_name": last_name, "dob": dob},
        )

        # Create the order
        order = Order.objects.create(
            patient=patient,
            provider=provider,
            primary_diagnosis=primary_diagnosis,
            medication_name=medication_name,
            additional_diagnoses=additional_diagnoses,
            weight_kg=weight_kg,
            allergies=allergies,
            medication_history=medication_history,
            patient_records=patient_records,
        )

        # Call LLM synchronously — user waits
        care_plan_text = generate_care_plan(order)
        order.care_plan = care_plan_text
        order.save()

        return redirect("care_plan", pk=order.pk)

    return render(request, "care/form.html")


def care_plan(request, pk):
    order = get_object_or_404(Order, pk=pk)
    return render(request, "care/care_plan.html", {"order": order})


def download_care_plan(request, pk):
    order = get_object_or_404(Order, pk=pk)
    filename = f"care_plan_{order.patient.mrn}_{order.medication_name.replace(' ', '_')}.txt"
    response = HttpResponse(order.care_plan, content_type="text/plain")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


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
