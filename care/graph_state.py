"""LangGraph state definition for the care-plan generation workflow.

Every node reads from and writes to CarePlanState. Fields are Optional where
a node has not yet produced a value; nodes should only update keys they own.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from typing_extensions import TypedDict

from .generation_loop import HallucinationReport
from .router import RoutingStrategy


class WorkflowStatus(str, Enum):
    PENDING = "pending"           # not yet started
    GENERATING = "generating"     # LLM call in progress
    VERIFYING = "verifying"       # hallucination check in progress
    REPAIRING = "repairing"       # auto-correct in progress
    FORMAT_ERROR = "format_error" # LLM output failed schema/format check
    ESCALATED = "escalated"       # high-risk finding → needs pharmacist
    SUCCESS = "success"
    FAILED = "failed"             # exhausted all retries


class ErrorRecord(TypedDict):
    step: str          # "generate" | "verify" | "format"
    attempt: int
    error_type: str    # "hallucination" | "format" | "parse"
    detail: str        # human-readable description
    prompt_version: str
    model: str


class CarePlanState(TypedDict):
    # ------------------------------------------------------------------
    # Input — set once at graph entry, never mutated
    # ------------------------------------------------------------------
    order_id: int
    patient_data: dict[str, Any]     # demographics, diagnoses, medications, etc.
    initial_prompt: str              # base prompt text before any feedback

    # ------------------------------------------------------------------
    # Routing — written by the router node
    # ------------------------------------------------------------------
    routing_strategy: Optional[RoutingStrategy]

    # ------------------------------------------------------------------
    # Generation — written by the generate node
    # ------------------------------------------------------------------
    current_prompt_version: str      # e.g. "v1", "v2", "v3"
    current_model: str               # e.g. "claude-sonnet-4-6", "gpt-5.5"
    llm_output: Optional[str]        # raw text from the LLM
    parsed_output: Optional[dict[str, Any]]  # validated/parsed care plan

    # ------------------------------------------------------------------
    # Verification — written by the verify node
    # ------------------------------------------------------------------
    verification_report: Optional[HallucinationReport]

    # ------------------------------------------------------------------
    # Retry tracking — updated by generate and repair nodes
    # ------------------------------------------------------------------
    generate_retry_count: int        # how many times generate has been called
    format_retry_count: int          # how many prompt versions have been tried
    repair_retry_count: int          # how many hallucination repairs attempted

    # ------------------------------------------------------------------
    # Error history — appended to by any failing node
    # All prior errors are injected into the next generation prompt so the
    # model knows what mistakes to avoid.
    # ------------------------------------------------------------------
    error_log: list[ErrorRecord]
    accumulated_feedback: str        # built-up prompt suffix from error_log

    # ------------------------------------------------------------------
    # Workflow control
    # ------------------------------------------------------------------
    status: WorkflowStatus
    escalate_to_pharmacist: bool     # set True by verify if risk_level="high"
    final_output: Optional[str]      # the accepted care plan text
