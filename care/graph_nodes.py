"""LangGraph node implementations for the care-plan generation workflow.

Each node is a callable (state: CarePlanState) -> dict that reads from the
shared state, calls exactly one service, and returns only the keys it changed.
LangGraph merges the returned dict into the full state automatically.

Nodes with external dependencies (LLM client, evaluator) are created via
make_* factory functions so the inner function can be unit-tested by passing
a fake client or evaluator without touching LangGraph at all.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .care_plan_schema import validate_care_plan
from .generation_loop import Evaluator
from .graph_state import CarePlanState, ErrorRecord, WorkflowStatus
from .router import assess_complexity

# Ordered list of prompt versions to cycle through on format failures.
# The router picks the starting version; each format failure advances by one.
PROMPT_VERSION_SEQUENCE = ("v1", "v2", "v3")

MAX_FORMAT_RETRIES = 3
MAX_REPAIR_RETRIES = 2


# ---------------------------------------------------------------------------
# route_node — no I/O, no dependencies
# ---------------------------------------------------------------------------

def route_node(state: CarePlanState) -> dict:
    """Assess patient complexity and select the initial prompt version + model.

    Simple patients (≤1 diagnosis, ≤2 medications, no allergy history) route
    to the lighter prompt and model; everything else routes to the stronger
    complex path. See router.assess_complexity for the full rules.
    """
    strategy = assess_complexity(state["patient_data"])
    return {
        "routing_strategy": strategy,
        "current_prompt_version": strategy.prompt_version,
        "current_model": strategy.model,
        "status": WorkflowStatus.GENERATING,
    }


# ---------------------------------------------------------------------------
# generate_node factory
# ---------------------------------------------------------------------------

def make_generate_node(
    *,
    provider: str = "anthropic",
    client: Any | None = None,
) -> Callable[[CarePlanState], dict]:
    """Return a generate node bound to the given LLM provider and client.

    On success  → sets llm_output, parsed_output, status=VERIFYING.
    On StructuredOutputError → increments format_retry_count, appends to
    error_log, rebuilds accumulated_feedback, status=FORMAT_ERROR.
    The graph's conditional edge then decides whether to retry (advancing the
    prompt version) or give up.
    """

    def generate_node(state: CarePlanState) -> dict:
        # Lazy import keeps this module importable without Django at load time.
        from .structured_llm import (  # noqa: PLC0415
            StructuredOutputError,
            generate_structured_care_plan,
        )

        attempt = state["generate_retry_count"] + 1

        try:
            plan = generate_structured_care_plan(
                provider=provider,
                patient_record=_format_patient_record(state["patient_data"]),
                reference_material=state["initial_prompt"],
                model=state["current_model"],
                prompt_version=state["current_prompt_version"],
                repair_instructions=state["accumulated_feedback"] or None,
                client=client,
            )
            return {
                "llm_output": plan.model_dump_json(),
                "parsed_output": plan.model_dump(mode="json"),
                "generate_retry_count": attempt,
                "status": WorkflowStatus.VERIFYING,
            }

        except StructuredOutputError as exc:
            error: ErrorRecord = {
                "step": "generate",
                "attempt": attempt,
                "error_type": "format",
                "detail": str(exc),
                "prompt_version": state["current_prompt_version"],
                "model": state["current_model"],
            }
            updated_log = state["error_log"] + [error]
            return {
                "llm_output": None,
                "parsed_output": None,
                "generate_retry_count": attempt,
                "format_retry_count": state["format_retry_count"] + 1,
                "current_prompt_version": _next_prompt_version(
                    state["current_prompt_version"]
                ),
                "error_log": updated_log,
                "accumulated_feedback": _build_feedback(updated_log),
                "status": WorkflowStatus.FORMAT_ERROR,
            }

    return generate_node


# ---------------------------------------------------------------------------
# format_check_node — raw-text path (non-structured-output providers)
# ---------------------------------------------------------------------------

def format_check_node(state: CarePlanState) -> dict:
    """Validate LLM output format and route accordingly.

    Three cases:
    1. parsed_output already set (structured-output path succeeded) → pass through.
    2. llm_output is None (structured-output path failed, already logged by
       generate_node) → pass through the FORMAT_ERROR so the edge can route back.
    3. llm_output present but unparsed (raw-text path) → validate here; on
       failure advance prompt version and append to error_log.
    """
    if state.get("parsed_output") is not None:
        return {"status": WorkflowStatus.VERIFYING}

    if not state.get("llm_output"):
        return {"status": WorkflowStatus.FORMAT_ERROR}

    result = validate_care_plan(state["llm_output"])

    if result.valid:
        return {
            "parsed_output": result.data,
            "status": WorkflowStatus.VERIFYING,
        }

    error: ErrorRecord = {
        "step": "format",
        "attempt": state["format_retry_count"] + 1,
        "error_type": "parse",
        "detail": "; ".join(
            f"{e.path}: {e.message}" for e in result.errors
        ),
        "prompt_version": state["current_prompt_version"],
        "model": state["current_model"],
    }
    updated_log = state["error_log"] + [error]
    return {
        "parsed_output": None,
        "format_retry_count": state["format_retry_count"] + 1,
        "current_prompt_version": _next_prompt_version(
            state["current_prompt_version"]
        ),
        "error_log": updated_log,
        "accumulated_feedback": _build_feedback(updated_log),
        "status": WorkflowStatus.FORMAT_ERROR,
    }


# ---------------------------------------------------------------------------
# verify_node factory
# ---------------------------------------------------------------------------

def make_verify_node(
    evaluator: Evaluator,
) -> Callable[[CarePlanState], dict]:
    """Return a verify node bound to the given Evaluator.

    Pass any Evaluator here — a SectioningVerifier for focused per-section
    checks, a VotingVerifier for multi-model consensus, or a simple
    rule-based evaluator for testing.

    On clean report  → sets final_output, status=SUCCESS.
    On high-risk     → sets escalate_to_pharmacist=True, status=ESCALATED.
    On low/medium    → logs findings, rebuilds feedback, status=REPAIRING.
                       The graph edge then routes back to generate_node so the
                       next attempt carries the hallucination detail in the prompt.
    """

    def verify_node(state: CarePlanState) -> dict:
        report = evaluator(state["llm_output"] or "")

        if report.valid:
            return {
                "verification_report": report,
                "final_output": state["llm_output"],
                "status": WorkflowStatus.SUCCESS,
            }

        if report.escalate_to_pharmacist or any(
            f.risk_level == "high" for f in report.findings
        ):
            return {
                "verification_report": report,
                "escalate_to_pharmacist": True,
                "status": WorkflowStatus.ESCALATED,
            }

        # Low / medium findings: feed back into next generation attempt
        error: ErrorRecord = {
            "step": "verify",
            "attempt": state["repair_retry_count"] + 1,
            "error_type": "hallucination",
            "detail": "; ".join(
                f"{f.path} [{f.risk_level}]: {f.message}"
                for f in report.findings
            ),
            "prompt_version": state["current_prompt_version"],
            "model": state["current_model"],
        }
        updated_log = state["error_log"] + [error]
        return {
            "verification_report": report,
            "repair_retry_count": state["repair_retry_count"] + 1,
            "error_log": updated_log,
            "accumulated_feedback": _build_feedback(updated_log),
            "status": WorkflowStatus.REPAIRING,
        }

    return verify_node


# ---------------------------------------------------------------------------
# auto_fix_node — deterministic patch for numerical hallucinations
# ---------------------------------------------------------------------------

def auto_fix_node(state: CarePlanState) -> dict:
    """Patch numerical fields in parsed_output using ground-truth patient_data.

    Called when verification found only numerical hallucinations (wrong weight,
    wrong ICD-10 code, wrong MRN). These are factual values we can correct
    directly from patient_data without another LLM call.

    Always sets status=SUCCESS — if the values are in patient_data, they are
    authoritative; remaining clinical reasoning is considered acceptable.
    """
    import copy
    import json as _json

    data = copy.deepcopy(state.get("parsed_output") or {})
    patient = state["patient_data"]
    ps = data.get("patient_summary", {})

    _FIXABLE = {
        "weight_kg": lambda v: float(v),
        "primary_diagnosis": lambda v: str(v),
        "mrn": lambda v: str(v),
    }
    for field, coerce in _FIXABLE.items():
        val = patient.get(field)
        if val is not None and field in ps:
            ps[field] = coerce(val)

    if "patient_summary" in data:
        data["patient_summary"] = ps

    fixed_json = _json.dumps(data, indent=2)
    return {
        "llm_output": fixed_json,
        "parsed_output": data,
        "final_output": fixed_json,
        "status": WorkflowStatus.SUCCESS,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_patient_record(patient_data: dict[str, Any]) -> str:
    return json.dumps(patient_data, indent=2, default=str)


def _next_prompt_version(current: str) -> str:
    """Return the next prompt version; stays at the last one if already there."""
    try:
        idx = PROMPT_VERSION_SEQUENCE.index(current)
        return PROMPT_VERSION_SEQUENCE[min(idx + 1, len(PROMPT_VERSION_SEQUENCE) - 1)]
    except ValueError:
        return PROMPT_VERSION_SEQUENCE[-1]


def _build_feedback(error_log: list[ErrorRecord]) -> str:
    """Summarise all prior failures into a prompt suffix for the next attempt."""
    if not error_log:
        return ""
    lines = [
        "Previous attempts had the following issues — do not repeat these mistakes:"
    ]
    for e in error_log:
        lines.append(
            f"- [{e['step']}, attempt {e['attempt']}] "
            f"{e['error_type']}: {e['detail']} "
            f"(prompt {e['prompt_version']}, model {e['model']})"
        )
    return "\n".join(lines)
