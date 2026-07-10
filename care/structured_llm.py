"""Provider adapters for schema-constrained care-plan generation."""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any, Literal, Protocol

import anthropic

from prompts import PromptManager

from .audit import AuditRecorder
from .care_plan_schema import CarePlan

if TYPE_CHECKING:
    from .models import Order

Provider = Literal["openai", "anthropic"]

DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"

class StructuredOutputError(RuntimeError):
    """Raised when an API does not return a complete parsed CarePlan."""


class OpenAIResponsesClient(Protocol):
    responses: Any


class AnthropicMessagesClient(Protocol):
    messages: Any


def build_care_plan_input(
    *,
    patient_record: str,
    reference_material: str,
    repair_instructions: str | None = None,
    prompt_version: str | None = None,
) -> str:
    prompt, _ = _render_care_plan_input(
        patient_record=patient_record,
        reference_material=reference_material,
        repair_instructions=repair_instructions,
        prompt_version=prompt_version,
    )
    return prompt


def _render_care_plan_input(
    *,
    patient_record: str,
    reference_material: str,
    repair_instructions: str | None = None,
    prompt_version: str | None = None,
) -> tuple[str, dict[str, str]]:
    rendered = PromptManager().render(
        workflow="careplan_structured",
        version=prompt_version,
        scenario=None if prompt_version else "structured_generation",
        variables={
            "patient_record": patient_record,
            "reference_material": reference_material,
        },
    )
    prompt = rendered.content
    if repair_instructions:
        prompt = f"{prompt}\n\n{repair_instructions}"
    return prompt, rendered.metadata()


def generate_openai_care_plan(
    *,
    patient_record: str,
    reference_material: str,
    client: OpenAIResponsesClient | None = None,
    model: str | None = None,
    repair_instructions: str | None = None,
    prompt_version: str | None = None,
    audit_enabled: bool = False,
    audit_order: "Order | None" = None,
    audit_run_group_id: uuid.UUID | None = None,
    audit_attempt_number: int = 1,
) -> CarePlan:
    """Generate and parse a CarePlan with OpenAI Structured Outputs."""
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise StructuredOutputError(
                "The openai package is required for the OpenAI provider."
            ) from exc
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    selected_model = model or os.environ.get(
        "OPENAI_MODEL", DEFAULT_OPENAI_MODEL
    )
    prompt, prompt_metadata = _render_care_plan_input(
        patient_record=patient_record,
        reference_material=reference_material,
        repair_instructions=repair_instructions,
        prompt_version=prompt_version,
    )
    recorder = _start_structured_audit(
        enabled=audit_enabled,
        workflow="structured_careplan_generation",
        provider="openai",
        model=selected_model,
        prompt=prompt,
        prompt_metadata=prompt_metadata,
        input_snapshot={
            "patient_record": patient_record,
            "reference_material": reference_material,
            "repair_instructions": repair_instructions,
        },
        model_parameters={"text_format": "CarePlan"},
        order=audit_order,
        run_group_id=audit_run_group_id,
        attempt_number=audit_attempt_number,
    )

    try:
        response = client.responses.parse(
            model=selected_model,
            input=[{"role": "user", "content": prompt}],
            text_format=CarePlan,
        )

        status = getattr(response, "status", None)
        raw_output = getattr(response, "output_text", "") or ""
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            error = StructuredOutputError(
                f"OpenAI response was incomplete: {details}"
            )
            _audit_failure(
                recorder,
                error,
                response=response,
                raw_output=raw_output,
                status="truncated",
                parse_success=False,
            )
            raise error

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            refusal = _find_openai_refusal(response)
            if refusal:
                error = StructuredOutputError(
                    f"OpenAI refused the request: {refusal}"
                )
                audit_status = "refused"
            else:
                error = StructuredOutputError(
                    "OpenAI returned no parsed care-plan output."
                )
                audit_status = "parse_failed"
            _audit_failure(
                recorder,
                error,
                response=response,
                raw_output=raw_output,
                status=audit_status,
                parse_success=False,
            )
            raise error

        plan = _ensure_care_plan(parsed)
        _audit_success(
            recorder,
            response=response,
            raw_output=raw_output
            or plan.model_dump_json(),
            stop_reason=status or "",
            parse_success=True,
        )
        return plan
    except Exception as exc:
        if recorder and not _audit_is_finalized(recorder):
            recorder.fail(exc)
        raise


def generate_anthropic_care_plan(
    *,
    patient_record: str,
    reference_material: str,
    client: AnthropicMessagesClient | None = None,
    model: str | None = None,
    max_tokens: int = 4096,
    repair_instructions: str | None = None,
    prompt_version: str | None = None,
    audit_enabled: bool = False,
    audit_order: "Order | None" = None,
    audit_run_group_id: uuid.UUID | None = None,
    audit_attempt_number: int = 1,
) -> CarePlan:
    """Generate and parse a CarePlan with Claude Structured Outputs."""
    if client is None:
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    selected_model = model or os.environ.get(
        "ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL
    )
    prompt, prompt_metadata = _render_care_plan_input(
        patient_record=patient_record,
        reference_material=reference_material,
        repair_instructions=repair_instructions,
        prompt_version=prompt_version,
    )
    recorder = _start_structured_audit(
        enabled=audit_enabled,
        workflow="structured_careplan_generation",
        provider="anthropic",
        model=selected_model,
        prompt=prompt,
        prompt_metadata=prompt_metadata,
        input_snapshot={
            "patient_record": patient_record,
            "reference_material": reference_material,
            "repair_instructions": repair_instructions,
        },
        model_parameters={
            "max_tokens": max_tokens,
            "output_format": "CarePlan",
        },
        order=audit_order,
        run_group_id=audit_run_group_id,
        attempt_number=audit_attempt_number,
    )

    try:
        response = client.messages.parse(
            model=selected_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_format=CarePlan,
        )

        stop_reason = getattr(response, "stop_reason", None)
        raw_output = _anthropic_output_text(response)
        if stop_reason == "refusal":
            error = StructuredOutputError("Claude refused the request.")
            _audit_failure(
                recorder,
                error,
                response=response,
                raw_output=raw_output,
                status="refused",
                stop_reason=stop_reason,
                parse_success=False,
            )
            raise error
        if stop_reason == "max_tokens":
            error = StructuredOutputError(
                "Claude output was truncated; increase max_tokens and retry."
            )
            _audit_failure(
                recorder,
                error,
                response=response,
                raw_output=raw_output,
                status="truncated",
                stop_reason=stop_reason,
                parse_success=False,
            )
            raise error

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            error = StructuredOutputError(
                "Claude returned no parsed care-plan output."
            )
            _audit_failure(
                recorder,
                error,
                response=response,
                raw_output=raw_output,
                status="parse_failed",
                stop_reason=stop_reason or "",
                parse_success=False,
            )
            raise error

        plan = _ensure_care_plan(parsed)
        _audit_success(
            recorder,
            response=response,
            raw_output=raw_output or plan.model_dump_json(),
            stop_reason=stop_reason or "",
            parse_success=True,
        )
        return plan
    except Exception as exc:
        if recorder and not _audit_is_finalized(recorder):
            recorder.fail(exc)
        raise


def generate_structured_care_plan(
    *,
    provider: Provider,
    patient_record: str,
    reference_material: str,
    client: Any | None = None,
    model: str | None = None,
    repair_instructions: str | None = None,
    prompt_version: str | None = None,
    audit_enabled: bool = False,
    audit_order: "Order | None" = None,
    audit_run_group_id: uuid.UUID | None = None,
    audit_attempt_number: int = 1,
) -> CarePlan:
    """Provider-neutral entry point returning the same CarePlan model."""
    if provider == "openai":
        return generate_openai_care_plan(
            patient_record=patient_record,
            reference_material=reference_material,
            client=client,
            model=model,
            repair_instructions=repair_instructions,
            prompt_version=prompt_version,
            audit_enabled=audit_enabled,
            audit_order=audit_order,
            audit_run_group_id=audit_run_group_id,
            audit_attempt_number=audit_attempt_number,
        )
    if provider == "anthropic":
        return generate_anthropic_care_plan(
            patient_record=patient_record,
            reference_material=reference_material,
            client=client,
            model=model,
            repair_instructions=repair_instructions,
            prompt_version=prompt_version,
            audit_enabled=audit_enabled,
            audit_order=audit_order,
            audit_run_group_id=audit_run_group_id,
            audit_attempt_number=audit_attempt_number,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def _ensure_care_plan(parsed: Any) -> CarePlan:
    if isinstance(parsed, CarePlan):
        return parsed
    return CarePlan.model_validate(parsed)


def _find_openai_refusal(response: Any) -> str | None:
    for output_item in getattr(response, "output", []) or []:
        for content_item in getattr(output_item, "content", []) or []:
            refusal = getattr(content_item, "refusal", None)
            if refusal:
                return str(refusal)
    return None


def _start_structured_audit(
    *,
    enabled: bool,
    workflow: str,
    provider: str,
    model: str,
    prompt: str,
    prompt_metadata: dict[str, str],
    input_snapshot: dict[str, Any],
    model_parameters: dict[str, Any],
    order: "Order | None",
    run_group_id: uuid.UUID | None,
    attempt_number: int,
) -> AuditRecorder | None:
    if not enabled:
        return None
    return AuditRecorder.start(
        workflow=workflow,
        provider=provider,
        model=model,
        rendered_prompt=prompt,
        input_snapshot=input_snapshot,
        model_parameters=model_parameters,
        prompt_metadata=prompt_metadata,
        order=order,
        run_group_id=run_group_id,
        attempt_number=attempt_number,
    )


def _audit_success(
    recorder: AuditRecorder | None,
    *,
    response: Any,
    raw_output: str,
    stop_reason: str,
    parse_success: bool,
) -> None:
    if not recorder:
        return
    recorder.succeed(
        raw_output=raw_output,
        raw_response=response,
        token_usage=getattr(response, "usage", None) or {},
        provider_request_id=str(getattr(response, "id", "") or ""),
        stop_reason=stop_reason,
        parse_success=parse_success,
    )


def _audit_failure(
    recorder: AuditRecorder | None,
    error: BaseException,
    *,
    response: Any,
    raw_output: str,
    status: str,
    stop_reason: str = "",
    parse_success: bool,
) -> None:
    if not recorder:
        return
    recorder.fail(
        error,
        raw_output=raw_output,
        raw_response=response,
        status=status,
        stop_reason=stop_reason,
        parse_success=parse_success,
        token_usage=getattr(response, "usage", None) or {},
        provider_request_id=str(getattr(response, "id", "") or ""),
    )


def _audit_is_finalized(recorder: AuditRecorder) -> bool:
    from .models import GenerationAudit

    return GenerationAudit.objects.filter(
        pk=recorder.audit_id,
        completed_at__isnull=False,
    ).exists()


def _anthropic_output_text(response: Any) -> str:
    return "".join(
        block.text
        for block in getattr(response, "content", []) or []
        if getattr(block, "type", None) == "text"
    )
