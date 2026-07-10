"""Conditional edge functions for the care-plan LangGraph workflow.

Each function receives the full state and returns a string key that LangGraph
uses to look up the next node. Keys map to node names or END in the graph
assembly (care_plan_graph.py).

Edge map:
  format_check ──► route_after_format ──► "verify" | "generate" | "end"
  verify       ──► route_after_verify ──► "end" | "auto_fix" | "generate"
"""

from __future__ import annotations

from .generation_loop import HallucinationFinding
from .graph_nodes import MAX_FORMAT_RETRIES, MAX_REPAIR_RETRIES
from .graph_state import CarePlanState, WorkflowStatus

# Paths that contain only factual/numerical values the auto_fix node can
# patch directly from patient_data. Everything else is a reasoning finding
# that requires regeneration.
_NUMERICAL_KEYWORDS: frozenset[str] = frozenset({
    "weight", "weight_kg",
    "primary_diagnosis", "diagnosis",
    "mrn", "dob", "date_of_birth",
    "lab", "scr", "creatinine", "egfr",
})


# ---------------------------------------------------------------------------
# After format_check
# ---------------------------------------------------------------------------

def route_after_format(state: CarePlanState) -> str:
    """Three outcomes after format_check_node:

    parse success     → "verify"
    parse failed, retries remaining → "generate"  (next prompt version)
    parse failed, retries exhausted → "end"        (FAILED)
    """
    if state["status"] == WorkflowStatus.VERIFYING:
        return "verify"

    # FORMAT_ERROR — check whether we have retries left
    if state["format_retry_count"] < MAX_FORMAT_RETRIES:
        return "generate"

    return "end"


# ---------------------------------------------------------------------------
# After verify
# ---------------------------------------------------------------------------

def route_after_verify(state: CarePlanState) -> str:
    """Four outcomes after verify_node:

    no hallucination              → "end"       (SUCCESS)
    high-risk finding             → "end"       (ESCALATED — pharmacist handles)
    low/medium, all numerical     → "auto_fix"  (patch values, no LLM call)
    low/medium, any reasoning     → "generate"  (regenerate with full error context)
    repair retries exhausted      → "end"       (FAILED)
    """
    status = state["status"]

    if status == WorkflowStatus.SUCCESS:
        return "end"

    if status == WorkflowStatus.ESCALATED:
        return "end"

    # REPAIRING — classify findings to decide the repair strategy
    report = state.get("verification_report")
    if report and report.findings:
        if _all_numerical(report.findings):
            return "auto_fix"

    # Any reasoning finding or empty findings: regenerate
    if state["repair_retry_count"] < MAX_REPAIR_RETRIES:
        return "generate"

    return "end"


# ---------------------------------------------------------------------------
# Finding classification
# ---------------------------------------------------------------------------

def _is_numerical(finding: HallucinationFinding) -> bool:
    """Return True if this finding can be auto-corrected from patient_data.

    A finding is numerical when its path references a factual patient field
    (weight, diagnosis code, MRN, lab values). Reasoning findings reference
    clinical sections (problems, goals, interventions, monitoring).
    """
    path_lower = finding.path.lower()
    return any(kw in path_lower for kw in _NUMERICAL_KEYWORDS)


def _all_numerical(findings: list[HallucinationFinding]) -> bool:
    return bool(findings) and all(_is_numerical(f) for f in findings)
