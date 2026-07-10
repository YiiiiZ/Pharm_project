"""
Care plan query system built with OpenAI Agents SDK.

Entry point: run `python -m care.query_agents` from the project root,
or import `triage_agent` and drive it with Runner directly.

Architecture:
  TriageAgent
    ├─ handoff → ProgressAgent   (progress / status queries)
    ├─ handoff → ReviewAgent     (verification report queries)
    └─ handoff → ResubmitAgent   (regeneration requests)

Each specialist agent hands back to TriageAgent for follow-up questions.
All tools are mocked — replace the bodies with real API calls.
"""

from __future__ import annotations

import asyncio
import random
import string
from datetime import datetime, timedelta

from agents import Agent, Runner, function_tool


# ---------------------------------------------------------------------------
# Mock tools — Progress Agent
# ---------------------------------------------------------------------------

@function_tool
def get_care_plan_status(patient_id: str) -> dict:
    """Return the current LangGraph node and run status for a care plan."""
    stages = ["Router", "Generation", "Verification", "Formatting"]
    stage = random.choice(stages)
    status = "running" if stage != "Formatting" else "completed"
    return {
        "patient_id": patient_id,
        "current_stage": stage,
        "status": status,
        "updated_at": datetime.now().isoformat(),
    }


@function_tool
def get_status_timeline(patient_id: str) -> dict:
    """Return timestamps for each completed stage."""
    base = datetime.now() - timedelta(minutes=15)
    return {
        "patient_id": patient_id,
        "timeline": [
            {"stage": "Router",     "started_at": (base).isoformat(),               "completed_at": (base + timedelta(minutes=1)).isoformat()},
            {"stage": "Generation", "started_at": (base + timedelta(minutes=1)).isoformat(), "completed_at": (base + timedelta(minutes=8)).isoformat()},
            {"stage": "Verification","started_at": (base + timedelta(minutes=8)).isoformat(), "completed_at": None},
        ],
    }


@function_tool
def get_estimated_completion(patient_id: str) -> dict:
    """Return estimated completion time based on historical averages."""
    eta = datetime.now() + timedelta(minutes=4)
    return {
        "patient_id": patient_id,
        "estimated_completion": eta.isoformat(),
        "confidence": "medium",
        "note": "Based on average Verification stage duration of 5 minutes.",
    }


# ---------------------------------------------------------------------------
# Mock tools — Review Agent
# ---------------------------------------------------------------------------

@function_tool
def get_verification_issues(patient_id: str) -> dict:
    """Return a summary list of flagged issues from the verification report."""
    return {
        "patient_id": patient_id,
        "issue_count": 2,
        "issues": [
            {"type": "drug_interaction", "severity": "high",   "summary": "Warfarin + Aspirin: increased bleeding risk"},
            {"type": "renal_function",   "severity": "medium", "summary": "Metformin dose may need adjustment (eGFR 45)"},
        ],
    }


@function_tool
def get_verification_report(patient_id: str) -> dict:
    """Return the full verification report for a care plan."""
    return {
        "patient_id": patient_id,
        "report_id": "vr-" + patient_id + "-001",
        "generated_at": datetime.now().isoformat(),
        "overall_result": "FAIL",
        "sections": {
            "drug_interaction": {
                "result": "FAIL",
                "details": "Warfarin (5mg) + Aspirin (100mg): clinically significant interaction. Risk of major bleeding. Recommend dose review or alternative.",
            },
            "renal_hepatic": {
                "result": "WARNING",
                "details": "eGFR 45 mL/min. Metformin 1000mg BID may accumulate. Consider reducing to 500mg BID.",
            },
            "allergy_risk": {
                "result": "PASS",
                "details": "No known allergen conflicts detected.",
            },
        },
    }


@function_tool
def get_sub_report(patient_id: str, report_type: str) -> dict:
    """
    Return a single sub-report section.
    report_type: one of 'drug_interaction', 'renal_hepatic', 'allergy_risk'
    """
    reports = {
        "drug_interaction": {
            "result": "FAIL",
            "details": "Warfarin (5mg) + Aspirin (100mg): clinically significant interaction.",
            "recommendation": "Review anticoagulation strategy with prescribing physician.",
        },
        "renal_hepatic": {
            "result": "WARNING",
            "details": "eGFR 45 — Metformin accumulation risk.",
            "recommendation": "Reduce Metformin to 500mg BID and recheck eGFR in 3 months.",
        },
        "allergy_risk": {
            "result": "PASS",
            "details": "No allergen conflicts.",
            "recommendation": "No action required.",
        },
    }
    section = reports.get(report_type, {"error": f"Unknown report type: {report_type}"})
    return {"patient_id": patient_id, "report_type": report_type, **section}


# ---------------------------------------------------------------------------
# Mock tools — Resubmit Agent
# ---------------------------------------------------------------------------

@function_tool
def check_regeneration_eligibility(patient_id: str) -> dict:
    """Check whether a care plan can be submitted for regeneration."""
    return {
        "patient_id": patient_id,
        "eligible": True,
        "reason": "Previous run completed (FAIL). Regeneration allowed.",
        "active_job_id": None,
    }


@function_tool
def submit_regeneration(patient_id: str, reason: str) -> dict:
    """Trigger a new LangGraph workflow run for the given patient. Returns job_id immediately (async)."""
    job_id = "regen-" + "".join(random.choices(string.digits, k=6))
    return {
        "patient_id": patient_id,
        "job_id": job_id,
        "status": "queued",
        "submitted_at": datetime.now().isoformat(),
        "reason_recorded": reason,
        "message": f"Regeneration queued. Use job_id '{job_id}' to track progress.",
    }


@function_tool
def get_regeneration_status(job_id: str) -> dict:
    """Check the status of a previously submitted regeneration job."""
    return {
        "job_id": job_id,
        "status": "running",
        "current_stage": "Generation",
        "started_at": (datetime.now() - timedelta(minutes=2)).isoformat(),
        "estimated_completion": (datetime.now() + timedelta(minutes=5)).isoformat(),
    }


# ---------------------------------------------------------------------------
# Agent definitions
#
# Circular reference: TriageAgent ↔ specialist agents.
# Strategy: define specialists first without the back-handoff,
# then build TriageAgent, then patch back-handoffs.
# ---------------------------------------------------------------------------

_TRIAGE_INSTRUCTIONS = """
You are the Triage Agent for a pharmacy care plan system.

Your ONLY action is to call a handoff tool. You never answer questions.
You never describe what you are about to do. You never explain your reasoning.
You just call the handoff tool, immediately, silently.

== Step 1: Extract patient ID ==
Treat ANY number or alphanumeric token in the message as the patient/order ID.
These all contain a valid ID — extract and pass it to the specialist:
  "订单 12345"  → 12345
  "P-1042 的报告" → P-1042
  "case #99"    → 99
  "帮我看看 ABC-7" → ABC-7
Only ask for an ID when the message contains zero numbers and zero alphanumeric codes.

== Step 2: Classify intent and call handoff ==

→ ProgressAgent
  Use when the pharmacist asks about pipeline stage, run status, or timing.
  Examples:
    "订单 12345 的 care plan 生成到哪了"
    "P-99 做完了吗"
    "12345 卡住了吗"
    "还需要多久"
  Keywords: 进度、到哪了、到哪一步、完成了吗、多久、状态、卡住、生成中

→ ReviewAgent
  Use when the pharmacist asks why something was flagged, or wants audit details.
  Examples:
    "为什么 12345 的 care plan 标了 hallucination"
    "P-1042 的 verification report 是什么"
    "12345 药物交互那条是什么意思"
    "审核没过，原因是什么"
  Keywords: 审核、报告、详情、为什么、标了、hallucination、verification、没过、原因、药物交互、过敏

→ ResubmitAgent
  Use when the pharmacist explicitly wants to regenerate or restart a care plan.
  Examples:
    "12345 的 care plan 有问题，请重新生成"
    "帮我重做 P-99 的方案"
    "重新跑一下 12345"
  Keywords: 重新生成、重做、重提、重跑、再生成、restart、resubmit

== Ambiguity ==
If and only if the intent cannot be determined after reading the full message,
ask ONE short clarifying question. Do not list options, just ask what they need.
"""

_PROGRESS_INSTRUCTIONS = """
You are the Progress Agent for a pharmacy care plan system.
You help pharmacists check where a care plan is in the generation pipeline:
  Router → Generation → Verification → Formatting

== Tool calling order ==
1. Always call get_care_plan_status first.
2. If the pharmacist asks about timing ("how long", "多久", "什么时候完成"):
   also call get_estimated_completion.
3. If the pharmacist asks about history or "why is it slow":
   call get_status_timeline.

== Responding ==
- State the current stage and status clearly.
- If status is 'failed': say which stage failed and suggest the pharmacist
  can request regeneration — but do NOT initiate it yourself.
- If status is 'completed': confirm it's done and ready.
- Keep replies concise; pharmacists are busy.

== Topic switch ==
If the pharmacist asks about verification details or wants to regenerate,
immediately hand off back to TriageAgent. Do not answer out of scope.
"""

_REVIEW_INSTRUCTIONS = """
You are the Review Agent for a pharmacy care plan system.
You help pharmacists understand verification reports and flagged issues.

== Tool calling order ==
1. Always call get_verification_issues first — it gives the concise summary.
2. Only call get_verification_report if the pharmacist explicitly asks for
   the full report or all details.
3. If the pharmacist asks about one specific issue type, call get_sub_report
   with the matching type: 'drug_interaction', 'renal_hepatic', or 'allergy_risk'.

== Responding ==
- Lead with the issue count and severity (high/medium/low).
- For each flagged issue, state: what it is, why it's a problem, what to do.
- Do not invent clinical advice beyond what the tool returns.
- If the report shows no issues (all PASS), confirm clearly.

== Topic switch ==
If the pharmacist asks about pipeline progress or wants to regenerate,
immediately hand off back to TriageAgent. Do not answer out of scope.
"""

_RESUBMIT_INSTRUCTIONS = """
You are the Resubmit Agent for a pharmacy care plan system.
You handle care plan regeneration requests.

== Tool calling order ==
1. ALWAYS call check_regeneration_eligibility first — never skip this.
2. If eligible: call submit_regeneration with the patient_id and a brief reason
   extracted from what the pharmacist said.
3. If NOT eligible (active_job_id is present): do NOT call submit_regeneration.
   Tell the pharmacist a job is already running and give them the active job_id.
4. If the pharmacist asks about a job they already submitted, call
   get_regeneration_status with the job_id they provide.

== Submitting ==
- submit_regeneration is async — it returns a job_id immediately, not a result.
- After submitting, tell the pharmacist:
    - the job_id (so they can track progress)
    - that completion typically takes 5–10 minutes
    - they can check status by asking the Progress Agent

== Hard rules ==
- Never call submit_regeneration more than once per conversation turn.
- Never call submit_regeneration without first calling check_regeneration_eligibility.

== Topic switch ==
If the pharmacist asks about pipeline progress or verification details,
immediately hand off back to TriageAgent. Do not answer out of scope.
"""


# Build specialist agents first (no back-handoff yet)
progress_agent = Agent(
    name="ProgressAgent",
    instructions=_PROGRESS_INSTRUCTIONS,
    tools=[get_care_plan_status, get_status_timeline, get_estimated_completion],
)

review_agent = Agent(
    name="ReviewAgent",
    instructions=_REVIEW_INSTRUCTIONS,
    tools=[get_verification_issues, get_verification_report, get_sub_report],
)

resubmit_agent = Agent(
    name="ResubmitAgent",
    instructions=_RESUBMIT_INSTRUCTIONS,
    tools=[check_regeneration_eligibility, submit_regeneration, get_regeneration_status],
)

# Build Triage with handoffs to all three specialists
triage_agent = Agent(
    name="TriageAgent",
    instructions=_TRIAGE_INSTRUCTIONS,
    handoffs=[progress_agent, review_agent, resubmit_agent],
)

# Patch back-handoffs so specialists can return control to Triage
progress_agent.handoffs.append(triage_agent)
review_agent.handoffs.append(triage_agent)
resubmit_agent.handoffs.append(triage_agent)


# ---------------------------------------------------------------------------
# CLI entry point — demo queries
# ---------------------------------------------------------------------------

async def _demo() -> None:
    queries = [
        "Patient P-1042 的 care plan 现在到哪一步了？",
        "P-2089 的 verification report 有什么问题？",
        "P-3301 的 care plan 有问题，帮我重新生成一下，原因是药物交互审核未通过。",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print(f"Pharmacist: {query}")
        print("=" * 60)
        result = await Runner.run(triage_agent, query)
        print(f"System: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(_demo())
