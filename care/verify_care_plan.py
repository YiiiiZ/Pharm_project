"""
Layer 1 care plan verification — pure code, no AI.

Checks:
  1. Numeric values in the care plan match the order record (weight, diagnosis code).
  2. Drug-like terms in the care plan are grounded in at least one authorized source
     field (medication_name, medication_history, patient_records, allergies).

Usage (Django shell or management command):
    from care.verify_care_plan import verify_order
    report = verify_order(order_id=42)
    print(report)

Usage (standalone, set DJANGO_SETTINGS_MODULE first):
    python care/verify_care_plan.py <order_id>
"""

import re
import sys
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

MATCH = "MATCH"
MISMATCH = "MISMATCH"
NOT_IN_RECORD = "NOT_IN_RECORD"


@dataclass
class CheckResult:
    check: str          # human-readable check name
    status: str         # MATCH | MISMATCH | NOT_IN_RECORD
    detail: str         # explanation
    expected: str = ""  # value from DB
    found: str = ""     # value extracted from care plan


@dataclass
class VerificationReport:
    order_id: int
    results: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.status == MATCH for r in self.results)

    def __str__(self) -> str:
        lines = [f"=== Verification Report — Order #{self.order_id} ==="]
        for r in self.results:
            lines.append(f"[{r.status:<15}] {r.check}: {r.detail}")
        lines.append(f"--- {'PASS' if self.passed else 'FAIL'} ---")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def verify_order(order_id: int) -> VerificationReport:
    from care.models import Order  # imported here to keep module importable outside Django

    order = Order.objects.select_related("patient", "provider").get(pk=order_id)
    report = VerificationReport(order_id=order_id)

    if not order.care_plan:
        report.results.append(CheckResult("care_plan", MATCH, "No care plan to verify"))
        return report

    report.results.extend(_check_numerics(order))
    report.results.extend(_check_drug_names(order))
    return report


# ---------------------------------------------------------------------------
# Check 1: numeric values
# ---------------------------------------------------------------------------

def _check_numerics(order) -> List[CheckResult]:
    results = []
    care_plan_lower = order.care_plan.lower()

    # Weight
    if order.weight_kg:
        results.append(_check_weight(order.weight_kg, care_plan_lower))

    # Primary diagnosis ICD-10 code
    if order.primary_diagnosis:
        results.append(_check_icd10(order.primary_diagnosis, order.care_plan))

    return results


def _check_weight(weight_kg: str, care_plan_lower: str) -> CheckResult:
    try:
        expected = float(weight_kg.strip())
    except ValueError:
        return CheckResult("weight_kg", MATCH, f"Weight '{weight_kg}' is non-numeric, skipped")

    found_weights = [
        float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*kg", care_plan_lower)
    ]

    if not found_weights:
        return CheckResult("weight_kg", MATCH, "Weight not mentioned in care plan", str(expected))

    mismatches = [w for w in found_weights if abs(w - expected) > 0.5]
    if not mismatches:
        return CheckResult(
            "weight_kg", MATCH,
            f"All weight mentions match record ({expected} kg)",
            str(expected), str(found_weights),
        )

    return CheckResult(
        "weight_kg", MISMATCH,
        f"Care plan has {mismatches} kg; record has {expected} kg",
        str(expected), str(mismatches),
    )


def _check_icd10(diagnosis: str, care_plan: str) -> CheckResult:
    code = diagnosis.strip().upper()
    if re.search(re.escape(code), care_plan, re.IGNORECASE):
        return CheckResult("primary_diagnosis", MATCH, f"ICD-10 code '{code}' present", code, code)
    return CheckResult(
        "primary_diagnosis", MISMATCH,
        f"ICD-10 code '{code}' not found in care plan",
        code, "",
    )


# ---------------------------------------------------------------------------
# Check 2: drug names not grounded in any authorized source
# ---------------------------------------------------------------------------

_COMMON_WORDS = {
    "the", "this", "these", "that", "with", "for", "and", "but", "not",
    "will", "has", "have", "had", "was", "were", "are", "is", "be",
    "plan", "goal", "goals", "list", "monitor", "patient", "pharmacist",
    "problem", "intervention", "monitoring", "section", "care", "clinical",
    "history", "dose", "daily", "therapy", "treatment", "adverse", "effect",
    "effects", "labs", "level", "levels", "renal", "hepatic", "function",
    "baseline", "follow", "education", "adherence", "side", "signs",
    "symptoms", "assessment", "review", "infusion", "injection", "oral",
    "weekly", "monthly", "annually", "provide", "ensure", "assess",
    "evaluate", "document", "obtain", "consider", "recommend", "refer",
}


def _extract_drug_candidates(text: str) -> List[str]:
    """
    Heuristic extraction of drug-like terms from free text.
    Targets:
      - Capitalized words not at sentence starts (e.g. Prednisone, Rituximab)
      - All-caps abbreviations of length >= 3 (e.g. IVIG, NSAID)
      - Any of the above optionally followed by a dose (e.g. Prednisone 20 mg)
    """
    dose_suffix = r"(?:\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|mL|units?|IU))?"

    # Capitalized words not at sentence/line start
    cap_pattern = re.compile(
        r"(?<![.!?\n])\s([A-Z][a-z]{2,}" + dose_suffix + r")"
    )
    # All-caps abbreviations
    abbrev_pattern = re.compile(r"\b([A-Z]{3,}" + dose_suffix + r")\b")

    candidates: set[str] = set()
    for m in cap_pattern.finditer(text):
        term = m.group(1).strip()
        word = term.split()[0].lower()
        if word not in _COMMON_WORDS:
            candidates.add(term)

    for m in abbrev_pattern.finditer(text):
        term = m.group(1).strip()
        word = term.split()[0].lower()
        if word not in _COMMON_WORDS:
            candidates.add(term)

    return sorted(candidates)


def _check_drug_names(order) -> List[CheckResult]:
    authorized = " ".join(
        filter(None, [
            order.medication_name,
            order.medication_history,
            order.patient_records,
            order.allergies,
        ])
    ).lower()

    candidates = _extract_drug_candidates(order.care_plan)
    if not candidates:
        return [CheckResult("drug_names", MATCH, "No drug candidates extracted from care plan")]

    results = []
    for term in candidates:
        base = term.split()[0]  # check just the drug name, not the dose part
        if base.lower() in authorized:
            results.append(CheckResult(
                "drug_name", MATCH,
                f"'{base}' grounded in source fields",
                base,
            ))
        else:
            results.append(CheckResult(
                "drug_name", NOT_IN_RECORD,
                f"'{term}' not found in medication_name, medication_history, patient_records, or allergies",
                "", term,
            ))

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import django
    import os

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pharm_project.settings")
    django.setup()

    if len(sys.argv) != 2:
        print("Usage: python care/verify_care_plan.py <order_id>")
        sys.exit(1)

    report = verify_order(int(sys.argv[1]))
    print(report)
    sys.exit(0 if report.passed else 1)
