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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts import PromptManager

# Load .env if present
env_path = PROJECT_ROOT / ".env"
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
    return build_rendered_prompt(case).content


def build_rendered_prompt(case: dict):
    inp = case["input"]
    patient = inp["patient"]
    provider = inp["provider"]
    order = inp["order"]

    dob = patient.get("dob", "Not provided")
    weight = order.get("weight_kg", "")
    weight_str = f"{weight} kg" if weight else "Not provided"
    return PromptManager().render(
        workflow="careplan_generation",
        scenario="evaluation",
        variables={
            "patient_name": (
                f"{patient.get('first_name', '')} "
                f"{patient.get('last_name', '')}"
            ).strip(),
            "mrn": patient.get("mrn", ""),
            "dob": dob,
            "sex": patient.get("sex", "Not provided"),
            "weight": weight_str,
            "allergies": order.get("allergies", "None known"),
            "provider_name": provider.get("name", ""),
            "provider_npi": provider.get("npi", ""),
            "medication_name": order.get("medication_name", ""),
            "primary_diagnosis": order.get("primary_diagnosis", ""),
            "primary_diagnosis_label": order.get(
                "primary_diagnosis_label", "Not provided"
            ),
            "additional_diagnoses": order.get(
                "additional_diagnoses", "None"
            ),
            "medication_history": order.get(
                "medication_history", "None provided"
            ),
            "patient_records": order.get(
                "patient_records", "None provided"
            ),
        },
    )


def run_case(client: anthropic.Anthropic, case: dict) -> dict:
    rendered_prompt = build_rendered_prompt(case)
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": rendered_prompt.content}],
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
        **rendered_prompt.metadata(),
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
