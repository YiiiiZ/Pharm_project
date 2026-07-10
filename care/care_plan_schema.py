"""Typed schema and validation helpers for LLM-generated care-plan JSON."""

from __future__ import annotations

import json
import re
from datetime import date
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)


class CarePlanModel(BaseModel):
    """Base configuration shared by every care-plan schema object."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PatientSummary(CarePlanModel):
    patient_name: str = Field(min_length=1)
    mrn: str = Field(min_length=1)
    date_of_birth: date | None = None
    weight_kg: float | None = Field(default=None, gt=0, le=1000)
    allergies: list[str]
    medication_name: str = Field(min_length=1)
    primary_diagnosis: str = Field(min_length=1)

    @field_validator("allergies", mode="before")
    @classmethod
    def coerce_allergies(cls, value: Any) -> Any:
        """Accept a single allergy string as a one-item list."""
        if isinstance(value, str):
            value = value.strip()
            return [value] if value else []
        return value


class Problem(CarePlanModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class Goal(CarePlanModel):
    description: str = Field(min_length=1)
    target: str = Field(min_length=1)
    timeframe: str | None = None
    related_problem: str | None = None
    source_ids: list[str] = Field(default_factory=list)


class Intervention(CarePlanModel):
    action: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    priority: Priority = Priority.MEDIUM
    monitoring: str | None = None
    requires_prescriber_confirmation: bool = False
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("requires_prescriber_confirmation", mode="before")
    @classmethod
    def coerce_boolean(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0"}:
                return False
        return value


class CarePlan(CarePlanModel):
    patient_summary: PatientSummary
    problems: list[Problem] = Field(min_length=1)
    goals: list[Goal] = Field(min_length=1)
    interventions: list[Intervention] = Field(min_length=1)


class FieldError(CarePlanModel):
    path: str
    message: str
    error_type: str
    input: Any = None


class CarePlanValidationResult(CarePlanModel):
    valid: bool
    data: dict[str, Any] | None = None
    errors: list[FieldError] = Field(default_factory=list)


def validate_care_plan(
    payload: str | bytes | bytearray | dict[str, Any],
) -> CarePlanValidationResult:
    """Validate raw LLM JSON and return normalized data or field-level errors.

    Pydantic performs safe coercions such as ``"72"`` to ``72.0``. A small
    pre-parser removes a Markdown JSON fence because models commonly emit one
    despite being asked for raw JSON.
    """
    try:
        if isinstance(payload, (str, bytes, bytearray)):
            raw_json = _remove_json_fence(payload)
            plan = CarePlan.model_validate_json(raw_json)
        else:
            plan = CarePlan.model_validate(payload)
    except ValidationError as exc:
        return CarePlanValidationResult(
            valid=False,
            errors=[_format_validation_error(error) for error in exc.errors()],
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return CarePlanValidationResult(
            valid=False,
            errors=[
                FieldError(
                    path="$",
                    message=f"Invalid JSON: {exc}",
                    error_type="json_invalid",
                    input=None,
                )
            ],
        )

    return CarePlanValidationResult(
        valid=True,
        data=plan.model_dump(mode="json"),
    )


def care_plan_json_schema() -> dict[str, Any]:
    """Return JSON Schema for prompts or provider structured-output APIs."""
    return CarePlan.model_json_schema()


def _remove_json_fence(payload: str | bytes | bytearray) -> str:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    text = payload.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return fenced.group(1) if fenced else text


def _format_validation_error(error: dict[str, Any]) -> FieldError:
    location = error.get("loc", ())
    path = "$" + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}"
        for part in location
    )
    return FieldError(
        path=path,
        message=error.get("msg", "Validation failed"),
        error_type=error.get("type", "validation_error"),
        input=error.get("input"),
    )
