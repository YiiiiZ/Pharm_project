#!/usr/bin/env python3
"""
Three end-to-end scenarios through the care-plan LangGraph pipeline.
All LLM calls are mocked — no API keys required.

Scenarios
─────────
  1. One-time pass      generate succeeds, verify clean → SUCCESS
  2. Parse failure      generate raises StructuredOutputError twice,
                        prompt version advances, succeeds on 3rd attempt
  3. Hallucination regen generate returns a plan with a fabricated reference,
                        verify flags it (reasoning/medium), regenerate with
                        accumulated feedback, verify clean → SUCCESS

Usage
─────
  python care/test_graph_scenarios.py        # standalone
  pytest  care/test_graph_scenarios.py -v   # pytest
"""

from __future__ import annotations

import sys
from pathlib import Path

# When run as `python care/test_graph_scenarios.py`, Python adds the script's
# own directory to sys.path, not the project root. Fix that first so both
# Django and our own imports can find the `care` package.
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# ── Minimal Django setup (needed for lazy imports inside generate_node) ──────
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        INSTALLED_APPS=["care"],
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    )
    django.setup()
# ─────────────────────────────────────────────────────────────────────────────

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

from care.care_plan_graph import build_care_plan_graph, make_initial_state
from care.care_plan_schema import CarePlan, Goal, Intervention, PatientSummary, Problem
from care.generation_loop import HallucinationFinding, HallucinationReport
from care.graph_state import WorkflowStatus
from care.structured_llm import StructuredOutputError

# ── Shared mock data ──────────────────────────────────────────────────────────

PATIENT_DATA = {
    "patient_name": "Sarah Chen",
    "mrn": "042891",
    "dob": "1978-03-12",
    "weight_kg": "58.3",
    "sex": "F",
    "allergies": ["Penicillin"],
    "diagnoses": ["G70.00", "N18.3"],
    "medications": ["IVIG", "Pyridostigmine", "Tacrolimus"],
    "medication_history": ["IVIG 2 g/kg q4w × 6 months"],
    "primary_diagnosis": "G70.00",
    "patient_records": "AChR antibody positive. CrCl 42 mL/min. Last infusion 4 weeks ago.",
}

GOOD_PLAN = CarePlan(
    patient_summary=PatientSummary(
        patient_name="Sarah Chen",
        mrn="042891",
        weight_kg=58.3,
        allergies=["Penicillin"],
        medication_name="IVIG",
        primary_diagnosis="G70.00",
    ),
    problems=[
        Problem(
            title="Myasthenia Gravis — AChR antibody positive",
            description=(
                "Autoimmune neuromuscular junction disorder with confirmed "
                "AChR antibody positivity requiring ongoing IVIG therapy."
            ),
            evidence=["AChR antibody positive", "Ongoing IVIG q4w"],
        )
    ],
    goals=[
        Goal(
            description="Maintain stable neuromuscular function",
            target="MG-ADL score improvement ≥ 3 points within 3 months",
            timeframe="3 months",
        )
    ],
    interventions=[
        Intervention(
            action="Administer IVIG 2 g/kg over 5 days per protocol",
            rationale="First-line immunotherapy for AChR+ myasthenia gravis",
        )
    ],
)

# Plan containing a fabricated guideline reference (reasoning hallucination)
HALLUCINATED_PLAN = CarePlan(
    patient_summary=PatientSummary(
        patient_name="Sarah Chen",
        mrn="042891",
        weight_kg=58.3,
        allergies=["Penicillin"],
        medication_name="IVIG",
        primary_diagnosis="G70.00",
    ),
    problems=[
        Problem(
            title="Myasthenia Gravis — AChR antibody positive",
            description="Autoimmune disorder managed per institutional MG protocol v3.",
            evidence=["AChR antibody positive"],
        )
    ],
    goals=[
        Goal(
            description="Reduce AChR antibody titre by 50%",
            target="50% titre reduction in 6 weeks",
            timeframe="6 weeks",
        )
    ],
    interventions=[
        Intervention(
            action="IVIG 2 g/kg + concurrent Rituximab 375 mg/m² monthly",
            rationale=(
                "Combined IVIG/Rituximab as per institutional guideline v3 "
                "(Appendix C, page 14)."   # fabricated — not in patient records
            ),
        )
    ],
)


# ── Evaluator factories ───────────────────────────────────────────────────────

def always_clean_evaluator(output: str) -> HallucinationReport:
    return HallucinationReport(valid=True, hallucination_count=0)


def hallucination_then_clean_evaluator():
    """First call finds a reasoning hallucination; second call is clean."""
    call_count = [0]

    def evaluator(output: str) -> HallucinationReport:
        call_count[0] += 1
        if call_count[0] == 1:
            return HallucinationReport(
                valid=False,
                hallucination_count=1,
                findings=[
                    HallucinationFinding(
                        path="$.interventions[0].rationale",
                        message=(
                            "References 'institutional guideline v3 Appendix C' — "
                            "not found in patient records or medication history"
                        ),
                        risk_level="medium",
                    )
                ],
            )
        return HallucinationReport(valid=True, hallucination_count=0)

    return evaluator


# ── State-change printer ──────────────────────────────────────────────────────

_TRUNC = 110


def _fmt(key: str, value: object) -> str:
    if value is None:
        return "None"
    if key == "routing_strategy" and value is not None:
        d = value.as_dict()
        return (
            f"complexity={d['complexity']}  "
            f"prompt={d['prompt_version']}  "
            f"model={d['model']}"
        )
    if key == "verification_report" and value is not None:
        r = value
        findings = [f"[{f.risk_level}] {f.path}" for f in r.findings]
        return (
            f"valid={r.valid}  "
            f"count={r.hallucination_count}  "
            f"findings={findings}"
        )
    if key == "error_log" and isinstance(value, list):
        if not value:
            return "[]"
        last = value[-1]
        return (
            f"[{len(value)} total]  last → "
            f"step={last['step']}  type={last['error_type']}  "
            f"prompt={last['prompt_version']}  "
            f"detail={last['detail'][:60]!r}"
        )
    if key in ("llm_output", "accumulated_feedback") and isinstance(value, str):
        short = value.replace("\n", " ")[:_TRUNC]
        return repr(short + ("…" if len(value) > _TRUNC else ""))
    if key == "parsed_output" and isinstance(value, dict):
        keys = list(value.keys())
        return f"{{sections: {keys}}}"
    if isinstance(value, str) and len(value) > _TRUNC:
        return repr(value[:_TRUNC] + "…")
    return repr(value)


def print_step(step: int, node: str, update: dict) -> None:
    print(f"\n  {'─' * 58}")
    print(f"  Step {step:>2}  │  {node.upper()}")
    print(f"  {'─' * 58}")
    for key, value in sorted(update.items()):
        label = f"  {key}"
        print(f"{label:<34} {_fmt(key, value)}")


# ── Scenario runner ───────────────────────────────────────────────────────────

def run_scenario(
    title: str,
    generate_side_effects: list,
    evaluator,
) -> None:
    print(f"\n\n  {'═' * 58}")
    print(f"  SCENARIO: {title}")
    print(f"  {'═' * 58}")

    compiled = build_care_plan_graph(evaluator=evaluator)
    initial = make_initial_state(
        order_id=1001,
        patient_data=PATIENT_DATA,
        initial_prompt=(
            "Generate a structured care plan for this specialty pharmacy order."
        ),
    )

    # Single stream pass: print each node update AND accumulate final state.
    # We do NOT call invoke() separately — stateful evaluators (with internal
    # call counters) would be consumed by the first pass and give wrong results
    # on a second run.
    current_state: dict = dict(initial)
    step = 0
    with patch(
        "care.structured_llm.generate_structured_care_plan",
        side_effect=generate_side_effects,
    ):
        for event in compiled.stream(initial, stream_mode="updates"):
            for node_name, update in event.items():
                step += 1
                print_step(step, node_name, update)
                current_state.update(update)

    final = current_state
    print(f"\n  {'─' * 58}")
    print(f"  RESULT")
    print(f"  {'─' * 58}")
    print(f"  status               {final['status']}")
    print(f"  prompt version used  {final['current_prompt_version']}")
    print(f"  model used           {final['current_model']}")
    print(f"  generate attempts    {final['generate_retry_count']}")
    print(f"  format retries       {final['format_retry_count']}")
    print(f"  repair retries       {final['repair_retry_count']}")
    print(f"  escalated            {final['escalate_to_pharmacist']}")
    print(f"  error log entries    {len(final['error_log'])}")
    if final.get("accumulated_feedback"):
        print()
        print("  accumulated_feedback →")
        for line in final["accumulated_feedback"].splitlines():
            print(f"    {line}")


# ── Diagram ───────────────────────────────────────────────────────────────────

def generate_diagram(out_dir: Path) -> None:
    from care.care_plan_graph import build_care_plan_graph

    compiled = build_care_plan_graph()
    mermaid = compiled.get_graph().draw_mermaid()

    md_path = out_dir / "care_plan_graph.md"
    md_path.write_text(f"```mermaid\n{mermaid}\n```\n", encoding="utf-8")
    print(f"\n  Mermaid diagram → {md_path}")

    try:
        png_bytes = compiled.get_graph().draw_mermaid_png()
        png_path = out_dir / "care_plan_graph.png"
        png_path.write_bytes(png_bytes)
        print(f"  Pipeline figure  → {png_path}")
    except Exception as exc:
        print(f"  PNG render skipped ({exc})")


# ── Tests (pytest-compatible) ─────────────────────────────────────────────────

def test_scenario_one_time_pass():
    compiled = build_care_plan_graph(evaluator=always_clean_evaluator)
    initial = make_initial_state(
        order_id=1,
        patient_data=PATIENT_DATA,
        initial_prompt="Generate care plan.",
    )
    with patch(
        "care.structured_llm.generate_structured_care_plan",
        return_value=GOOD_PLAN,
    ):
        final = compiled.invoke(initial)

    assert final["status"] == WorkflowStatus.SUCCESS
    assert final["format_retry_count"] == 0
    assert final["repair_retry_count"] == 0
    assert final["generate_retry_count"] == 1


def test_scenario_parse_failure_retry():
    compiled = build_care_plan_graph(evaluator=always_clean_evaluator)
    initial = make_initial_state(
        order_id=2,
        patient_data=PATIENT_DATA,
        initial_prompt="Generate care plan.",
    )
    with patch(
        "care.structured_llm.generate_structured_care_plan",
        side_effect=[
            StructuredOutputError("Output incomplete: missing goals section"),
            StructuredOutputError("Output incomplete: goals section empty"),
            GOOD_PLAN,
        ],
    ):
        final = compiled.invoke(initial)

    assert final["status"] == WorkflowStatus.SUCCESS
    assert final["format_retry_count"] == 2
    assert final["generate_retry_count"] == 3
    assert final["current_prompt_version"] == "v3"
    assert len(final["error_log"]) == 2


def test_scenario_hallucination_regen():
    evaluator = hallucination_then_clean_evaluator()
    compiled = build_care_plan_graph(evaluator=evaluator)
    initial = make_initial_state(
        order_id=3,
        patient_data=PATIENT_DATA,
        initial_prompt="Generate care plan.",
    )
    with patch(
        "care.structured_llm.generate_structured_care_plan",
        side_effect=[HALLUCINATED_PLAN, GOOD_PLAN],
    ):
        final = compiled.invoke(initial)

    assert final["status"] == WorkflowStatus.SUCCESS
    assert final["repair_retry_count"] == 1
    assert final["generate_retry_count"] == 2
    assert len(final["error_log"]) == 1
    assert final["error_log"][0]["error_type"] == "hallucination"
    assert "accumulated_feedback" in final
    assert "institutional guideline v3" in final["accumulated_feedback"]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent

    # ── Scenario 1: One-time pass ─────────────────────────────────────────────
    run_scenario(
        title="1 · One-Time Pass",
        generate_side_effects=[GOOD_PLAN],
        evaluator=always_clean_evaluator,
    )

    # ── Scenario 2: Parse failure × 2, then success ───────────────────────────
    run_scenario(
        title="2 · Parse Failure → Prompt Version Retry",
        generate_side_effects=[
            StructuredOutputError("Output incomplete: missing goals section"),
            StructuredOutputError("Output incomplete: goals section empty"),
            GOOD_PLAN,
        ],
        evaluator=always_clean_evaluator,
    )

    # ── Scenario 3: Hallucination detected → regenerate with feedback ─────────
    run_scenario(
        title="3 · Hallucination Detected → Regenerate with Feedback",
        generate_side_effects=[HALLUCINATED_PLAN, GOOD_PLAN],
        evaluator=hallucination_then_clean_evaluator(),
    )

    # ── Diagram ───────────────────────────────────────────────────────────────
    print(f"\n\n  {'═' * 58}")
    print("  PIPELINE DIAGRAM")
    print(f"  {'═' * 58}")
    generate_diagram(out_dir)
