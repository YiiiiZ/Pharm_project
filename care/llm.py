import anthropic
from django.conf import settings


def generate_care_plan(order) -> str:
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    dob_str = order.patient.dob.strftime("%Y-%m-%d") if order.patient.dob else "Not provided"

    prompt = f"""You are a clinical pharmacist at a specialty pharmacy generating a care plan for a patient order.

PATIENT INFORMATION
-------------------
Name: {order.patient.first_name} {order.patient.last_name}
MRN: {order.patient.mrn}
Date of Birth: {dob_str}
Weight: {order.weight_kg + " kg" if order.weight_kg else "Not provided"}
Allergies: {order.allergies if order.allergies else "None known"}
Referring Provider: {order.provider.name} (NPI: {order.provider.npi})

ORDER DETAILS
-------------
Medication: {order.medication_name}
Primary Diagnosis (ICD-10): {order.primary_diagnosis}
Additional Diagnoses: {order.additional_diagnoses if order.additional_diagnoses else "None"}
Medication History:
{order.medication_history if order.medication_history else "None provided"}

CLINICAL NOTES / PATIENT RECORDS
---------------------------------
{order.patient_records if order.patient_records else "None provided"}

Generate a clinical care plan with exactly these four labeled sections. Be specific, clinical, and actionable.

1. PROBLEM LIST
2. GOALS
3. PHARMACIST INTERVENTIONS
4. MONITORING PLAN
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
