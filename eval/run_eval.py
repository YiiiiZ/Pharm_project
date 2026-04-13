#!/usr/bin/env python3
"""
Run all eval cases through the LLM and record results in dataset.json.
Usage: python eval/run_eval.py
Requires ANTHROPIC_API_KEY in environment or .env file.
"""

import json
import os
import sys
import datetime
from pathlib import Path

# Load .env if present
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import anthropic

DATASET_PATH = Path(__file__).parent / "dataset.json"
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
RUN_TIMESTAMP = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")


def build_prompt(case: dict) -> str:
    inp = case["input"]
    patient = inp["patient"]
    provider = inp["provider"]
    order = inp["order"]

    dob = patient.get("dob", "Not provided")
    weight = order.get("weight_kg", "")
    weight_str = f"{weight} kg" if weight else "Not provided"

    return f"""You are a clinical pharmacist at a specialty pharmacy generating a care plan for a patient order.

PATIENT INFORMATION
-------------------
Name: {patient.get('first_name', '')} {patient.get('last_name', '')}
MRN: {patient.get('mrn', '')}
Date of Birth: {dob}
Sex: {patient.get('sex', 'Not provided')}
Weight: {weight_str}
Allergies: {order.get('allergies', 'None known')}
Referring Provider: {provider.get('name', '')} (NPI: {provider.get('npi', '')})

ORDER DETAILS
-------------
Medication: {order.get('medication_name', '')}
Primary Diagnosis (ICD-10): {order.get('primary_diagnosis', '')} — {order.get('primary_diagnosis_label', '')}
Additional Diagnoses: {order.get('additional_diagnoses', 'None')}
Medication History:
{order.get('medication_history', 'None provided')}

CLINICAL NOTES / PATIENT RECORDS
---------------------------------
{order.get('patient_records', 'None provided')}

Generate a clinical care plan with exactly these four labeled sections. Be specific, clinical, and actionable.

1. PROBLEM LIST
2. GOALS
3. PHARMACIST INTERVENTIONS
4. MONITORING PLAN
"""


def run_case(client: anthropic.Anthropic, case: dict) -> dict:
    prompt = build_prompt(case)
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    care_plan_text = message.content[0].text
    stop_reason = message.stop_reason

    # Determine next run_id
    existing_runs = case.get("runs", [])
    run_num = len(existing_runs) + 1
    run_id = f"run_{run_num:03d}"

    return {
        "run_id": run_id,
        "timestamp": RUN_TIMESTAMP,
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "stop_reason": stop_reason,
        "truncated": stop_reason == "max_tokens",
        "care_plan_text": care_plan_text,
        "evaluation": None,  # To be filled in manually
    }


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    data = json.loads(DATASET_PATH.read_text())

    for case in data["cases"]:
        case_id = case["id"]
        medication = case["metadata"]["medication"]
        print(f"\n{'='*60}")
        print(f"Running {case_id} — {medication}...")

        try:
            run_result = run_case(client, case)
            case.setdefault("runs", []).append(run_result)
            truncated = run_result["truncated"]
            print(f"  Done. stop_reason={run_result['stop_reason']}"
                  f"{' ⚠ TRUNCATED' if truncated else ''}")
            # Preview first 200 chars
            preview = run_result["care_plan_text"][:200].replace("\n", " ")
            print(f"  Preview: {preview}...")
        except Exception as e:
            print(f"  ERROR: {e}")

    DATASET_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\n{'='*60}")
    print(f"All runs saved to {DATASET_PATH}")


if __name__ == "__main__":
    main()
