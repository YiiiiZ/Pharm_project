"""Validation-feedback retry loop for generated care-plan JSON."""

from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import Any, Callable, Literal, Protocol

from pydantic import Field

from prompts import PromptManager

from .care_plan_schema import (
    CarePlanModel,
    CarePlanValidationResult,
    FieldError,
    validate_care_plan,
)

RawCarePlan = str | bytes | bytearray | dict[str, Any]


class RetryStatus(str, Enum):
    SUCCESS = "success"
    PARSE_FAILED = "parse_failed"


class CarePlanGenerator(Protocol):
    def __call__(
        self,
        *,
        attempt_number: int,
        repair_instructions: str | None,
    ) -> RawCarePlan:
        """Return one complete care-plan JSON candidate."""


Validator = Callable[[RawCarePlan], CarePlanValidationResult]


class GenerationAttempt(CarePlanModel):
    attempt_number: int
    raw_output: Any
    valid: bool
    errors: list[FieldError] = Field(default_factory=list)


class CarePlanRetryResult(CarePlanModel):
    status: RetryStatus
    attempts_used: int
    retries_used: int
    data: dict[str, Any] | None = None
    original_output: Any = None
    final_output: Any = None
    attempts: list[GenerationAttempt] = Field(default_factory=list)


def generate_with_validation_retry(
    generator: CarePlanGenerator,
    *,
    validator: Validator = validate_care_plan,
    max_retries: int = 2,
    audit_enabled: bool = False,
    audit_order: Any | None = None,
) -> CarePlanRetryResult:
    """Generate, validate, and repair a care plan.

    ``max_retries=2`` means at most three API calls: one initial attempt and
    two repair attempts. API/network exceptions are deliberately not converted
    into ``parse_failed`` because they require a separate transient-error retry
    policy and may not contain any model output to preserve.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater")

    attempts: list[GenerationAttempt] = []
    repair_instructions: str | None = None

    for attempt_number in range(1, max_retries + 2):
        raw_output = generator(
            attempt_number=attempt_number,
            repair_instructions=repair_instructions,
        )
        validation = validator(raw_output)
        attempts.append(
            GenerationAttempt(
                attempt_number=attempt_number,
                raw_output=_json_safe(raw_output),
                valid=validation.valid,
                errors=validation.errors,
            )
        )

        if validation.valid:
            return CarePlanRetryResult(
                status=RetryStatus.SUCCESS,
                attempts_used=attempt_number,
                retries_used=attempt_number - 1,
                data=validation.data,
                original_output=attempts[0].raw_output,
                final_output=attempts[-1].raw_output,
                attempts=attempts,
            )

        if attempt_number <= max_retries:
            repair_instructions = build_repair_instructions(
                raw_output=raw_output,
                errors=validation.errors,
            )

    return CarePlanRetryResult(
        status=RetryStatus.PARSE_FAILED,
        attempts_used=len(attempts),
        retries_used=max(0, len(attempts) - 1),
        original_output=attempts[0].raw_output if attempts else None,
        final_output=attempts[-1].raw_output if attempts else None,
        attempts=attempts,
    )


def generate_structured_care_plan_with_retry(
    *,
    provider: Literal["openai", "anthropic"],
    patient_record: str,
    reference_material: str,
    client: Any | None = None,
    model: str | None = None,
    validator: Validator = validate_care_plan,
    max_retries: int = 2,
) -> CarePlanRetryResult:
    """Run the retry loop using the project's structured-output adapters.

    Native structured-output APIs already enforce the JSON shape. This wrapper
    is most useful when ``validator`` also performs post-generation grounding
    or clinical checks and returns field-level errors in the same result type.
    """
    from .structured_llm import generate_structured_care_plan

    run_group_id = uuid.uuid4() if audit_enabled else None

    def generator(
        *,
        attempt_number: int,
        repair_instructions: str | None,
    ) -> dict[str, Any]:
        plan = generate_structured_care_plan(
            provider=provider,
            patient_record=patient_record,
            reference_material=reference_material,
            client=client,
            model=model,
            repair_instructions=repair_instructions,
            audit_enabled=audit_enabled,
            audit_order=audit_order,
            audit_run_group_id=run_group_id,
            audit_attempt_number=attempt_number,
        )
        return plan.model_dump(mode="json")

    return generate_with_validation_retry(
        generator,
        validator=validator,
        max_retries=max_retries,
    )


def build_repair_instructions(
    *,
    raw_output: RawCarePlan,
    errors: list[FieldError],
) -> str:
    """Render exact validation errors into a prompt for the next attempt."""
    return PromptManager().render(
        workflow="careplan_repair",
        scenario="careplan_repair",
        variables={
            "validation_errors": json.dumps(
                [error.model_dump(mode="json") for error in errors],
                indent=2,
                ensure_ascii=False,
            ),
            "previous_output": _raw_as_text(raw_output),
        },
    ).content


def _raw_as_text(raw_output: RawCarePlan) -> str:
    if isinstance(raw_output, (bytes, bytearray)):
        return raw_output.decode("utf-8", errors="replace")
    if isinstance(raw_output, str):
        return raw_output
    return json.dumps(raw_output, indent=2, ensure_ascii=False, default=str)


def _json_safe(raw_output: RawCarePlan) -> Any:
    if isinstance(raw_output, (bytes, bytearray)):
        return raw_output.decode("utf-8", errors="replace")
    return raw_output
