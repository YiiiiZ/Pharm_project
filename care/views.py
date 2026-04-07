from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse

from .models import Provider, Patient, Order
from .llm import generate_care_plan


def order_form(request):
    if request.method == "POST":
        # Pull form data
        npi = request.POST["provider_npi"].strip()
        provider_name = request.POST["provider_name"].strip()
        mrn = request.POST["mrn"].strip()
        first_name = request.POST["first_name"].strip()
        last_name = request.POST["last_name"].strip()
        primary_diagnosis = request.POST["primary_diagnosis"].strip()
        medication_name = request.POST["medication_name"].strip()
        additional_diagnoses = request.POST.get("additional_diagnoses", "").strip()
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
            defaults={"first_name": first_name, "last_name": last_name},
        )

        # Create the order
        order = Order.objects.create(
            patient=patient,
            provider=provider,
            primary_diagnosis=primary_diagnosis,
            medication_name=medication_name,
            additional_diagnoses=additional_diagnoses,
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
