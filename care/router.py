"""Pure-Python complexity router for care-plan generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_SIMPLE_PROMPT_VERSION = "v1"
DEFAULT_COMPLEX_PROMPT_VERSION = "v2"
DEFAULT_SIMPLE_MODEL = "gpt-5.5"
DEFAULT_COMPLEX_MODEL = "claude-sonnet-4-6"


@dataclass(frozen=True)
class RoutingStrategy:
    """Selected generation strategy for a patient record."""

    complexity: str
    prompt_version: str
    model: str
    decision_log: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "complexity": self.complexity,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "decision_log": list(self.decision_log),
        }


def assess_complexity(
    patient_record: Mapping[str, Any],
    *,
    simple_prompt_version: str = DEFAULT_SIMPLE_PROMPT_VERSION,
    complex_prompt_version: str = DEFAULT_COMPLEX_PROMPT_VERSION,
    simple_model: str = DEFAULT_SIMPLE_MODEL,
    complex_model: str = DEFAULT_COMPLEX_MODEL,
) -> RoutingStrategy:
    """Route a patient record to a generation strategy.

    Rules:
    - simple: diagnoses <= 1, medications <= 2, and no allergy history
    - complex: everything else
    """

    diagnoses = _get_items(patient_record, "diagnoses")
    medications = _get_items(patient_record, "medications")
    allergies = _get_items(patient_record, "allergies")
    labs = patient_record.get("lab_values") or patient_record.get("labs") or {}

    diagnosis_count = len(diagnoses)
    medication_count = len(medications)
    has_allergy_history = _has_allergy_history(allergies)

    is_simple = (
        diagnosis_count <= 1
        and medication_count <= 2
        and not has_allergy_history
    )

    decision_log = [
        (
            "diagnoses count="
            f"{diagnosis_count} (<= 1 required for simple)"
        ),
        (
            "medications count="
            f"{medication_count} (<= 2 required for simple)"
        ),
        (
            "allergy history="
            f"{'present' if has_allergy_history else 'absent'} "
            "(must be absent for simple)"
        ),
        f"labs provided={bool(labs)} (not used in routing decision)",
        (
            "decision="
            + ("simple" if is_simple else "complex")
        ),
    ]

    if is_simple:
        return RoutingStrategy(
            complexity="simple",
            prompt_version=simple_prompt_version,
            model=simple_model,
            decision_log=tuple(decision_log),
        )

    return RoutingStrategy(
        complexity="complex",
        prompt_version=complex_prompt_version,
        model=complex_model,
        decision_log=tuple(decision_log),
    )


def _get_items(record: Mapping[str, Any], key: str) -> list[Any]:
    value = record.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [] if not text else [text]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence):
        return [item for item in value if item not in (None, "", [])]
    return [value]


def _has_allergy_history(allergies: Sequence[Any]) -> bool:
    if not allergies:
        return False

    negations = {
        "none",
        "none known",
        "no known allergies",
        "no allergies",
        "nkda",
        "nka",
    }

    for allergy in allergies:
        if allergy is None:
            continue
        if isinstance(allergy, str):
            normalized = allergy.strip().lower()
            if not normalized or normalized in negations:
                continue
            return True
        return True

    return False
