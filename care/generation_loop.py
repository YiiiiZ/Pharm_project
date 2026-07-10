"""Pure-Python generator/evaluator retry loop for hallucination repair."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol


class LoopStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class GeneratedOutput:
    text: str
    cost: float = 0.0


@dataclass(frozen=True)
class HallucinationFinding:
    path: str
    message: str
    evidence: str = ""
    risk_level: str = "low"  # "low" | "medium" | "high"


@dataclass(frozen=True)
class HallucinationReport:
    valid: bool
    hallucination_count: int
    findings: list[HallucinationFinding] = field(default_factory=list)
    # Set by SectioningVerifier when any section finding carries risk_level="high"
    escalate_to_pharmacist: bool = False


@dataclass(frozen=True)
class RetryRoundLog:
    attempt_number: int
    hallucination_count_before: int
    hallucination_count_after: int
    hallucination_delta: int
    cost: float
    prompt_used: str
    output: str
    feedback_used: str = ""


@dataclass(frozen=True)
class RetryLoopResult:
    status: LoopStatus
    attempts_used: int
    retries_used: int
    original_output: str | None
    final_output: str | None
    total_cost: float
    decision_log: tuple[str, ...]
    attempts: tuple[RetryRoundLog, ...]
    final_report: HallucinationReport | None = None


class Generator(Protocol):
    def __call__(self, *, prompt: str, attempt_number: int) -> Any:
        """Generate one candidate response for the supplied prompt."""


class Evaluator(Protocol):
    def __call__(self, output: str) -> HallucinationReport:
        """Evaluate a generated response and report hallucination findings."""


FeedbackBuilder = Callable[[HallucinationReport], str]


def generate_with_retry_loop(
    generator: Generator,
    evaluator: Evaluator,
    *,
    initial_prompt: str,
    max_retries: int = 2,
    feedback_builder: FeedbackBuilder | None = None,
) -> RetryLoopResult:
    """Generate, evaluate, and repair with up to two retry rounds.

    The generator receives the current prompt. After each failed evaluation,
    the feedback text is appended to the prompt for the next attempt.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater")

    feedback_builder = feedback_builder or build_feedback_prompt
    prompt = initial_prompt
    total_cost = 0.0
    rounds: list[RetryRoundLog] = []
    original_output: str | None = None
    final_report: HallucinationReport | None = None

    for attempt_number in range(1, max_retries + 2):
        current_prompt = prompt
        output, cost = _normalize_generation_output(
            generator(prompt=current_prompt, attempt_number=attempt_number)
        )
        report = evaluator(output)

        if original_output is None:
            original_output = output

        hallucination_count_before = (
            rounds[-1].hallucination_count_after if rounds else 0
        )
        hallucination_count_after = report.hallucination_count
        hallucination_delta = (
            hallucination_count_after - hallucination_count_before
        )
        total_cost += cost

        feedback_used = ""
        if not report.valid and attempt_number <= max_retries:
            feedback_used = feedback_builder(report)
            prompt = f"{initial_prompt}\n\n{feedback_used}"

        rounds.append(
            RetryRoundLog(
                attempt_number=attempt_number,
                hallucination_count_before=hallucination_count_before,
                hallucination_count_after=hallucination_count_after,
                hallucination_delta=hallucination_delta,
                cost=cost,
                prompt_used=current_prompt,
                output=output,
                feedback_used=feedback_used,
            )
        )

        if report.valid:
            final_report = report
            return RetryLoopResult(
                status=LoopStatus.SUCCESS,
                attempts_used=attempt_number,
                retries_used=attempt_number - 1,
                original_output=original_output,
                final_output=output,
                total_cost=total_cost,
                decision_log=_build_decision_log(rounds),
                attempts=tuple(rounds),
                final_report=final_report,
            )

    final_report = report if "report" in locals() else None
    return RetryLoopResult(
        status=LoopStatus.FAILED,
        attempts_used=len(rounds),
        retries_used=max(0, len(rounds) - 1),
        original_output=original_output,
        final_output=rounds[-1].output if rounds else None,
        total_cost=total_cost,
        decision_log=_build_decision_log(rounds),
        attempts=tuple(rounds),
        final_report=final_report,
    )


def build_feedback_prompt(report: HallucinationReport) -> str:
    """Turn hallucination findings into a prompt fragment for the next round."""
    if report.valid or not report.findings:
        return "No hallucinations detected."

    lines = [
        "Previous output contained hallucinations. Revise the answer using the feedback below.",
        f"Hallucination count: {report.hallucination_count}",
    ]
    for finding in report.findings:
        parts = [f"- {finding.path}: {finding.message}"]
        if finding.evidence:
            parts.append(f"evidence={finding.evidence}")
        lines.append(" ".join(parts))
    lines.append("Return a complete replacement, not a patch.")
    return "\n".join(lines)


def _normalize_generation_output(value: Any) -> tuple[str, float]:
    if isinstance(value, GeneratedOutput):
        return value.text, float(value.cost)
    if isinstance(value, tuple) and len(value) == 2:
        text, cost = value
        return str(text), float(cost)
    if hasattr(value, "text") and hasattr(value, "cost"):
        return str(value.text), float(value.cost)
    return str(value), 0.0


def _build_decision_log(rounds: list[RetryRoundLog]) -> tuple[str, ...]:
    return tuple(
        (
            f"attempt={round_.attempt_number} "
            f"hallucinations {round_.hallucination_count_before}"
            f"->{round_.hallucination_count_after} "
            f"delta={round_.hallucination_delta} "
            f"cost={round_.cost:.4f}"
        )
        for round_ in rounds
    )
