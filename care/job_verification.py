"""Limited deterministic checks; these do not establish clinical correctness."""
import json
from .generation_loop import HallucinationFinding, HallucinationReport


def make_evaluator(patient, evidence):
    allowed_ids = {chunk["chunk_id"] for chunk in evidence}

    def evaluate(output):
        plan = json.loads(output)
        findings = []
        summary = plan["patient_summary"]
        for field in ("patient_name", "mrn", "date_of_birth", "weight_kg", "allergies", "medication_name", "primary_diagnosis"):
            if summary.get(field) != patient.get(field):
                findings.append(HallucinationFinding(
                    path=f"$.patient_summary.{field}",
                    message="Does not match the submitted patient record.",
                    evidence=json.dumps(patient.get(field)), risk_level="high",
                ))
        for section in ("problems", "goals", "interventions"):
            for index, item in enumerate(plan[section]):
                ids = item.get("source_ids", [])
                if not ids or not set(ids).issubset(allowed_ids):
                    findings.append(HallucinationFinding(
                        path=f"$.{section}[{index}].source_ids",
                        message="Every item must cite IDs from the supplied label evidence.",
                        risk_level="medium",
                    ))
        return HallucinationReport(
            valid=not findings, hallucination_count=len(findings), findings=findings,
            escalate_to_pharmacist=any(f.risk_level == "high" for f in findings),
        )
    return evaluate
