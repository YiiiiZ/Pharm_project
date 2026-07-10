"""Append-only audit recording for LLM generation attempts."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Mapping

from django.db import transaction
from django.utils import timezone

from .models import GenerationAudit, Order


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def snapshot_order(order: Order) -> dict[str, Any]:
    """Capture patient/order facts so later database edits cannot alter history."""
    return {
        "order_id": order.pk,
        "created_at": order.created_at,
        "patient": {
            "id": order.patient_id,
            "mrn": order.patient.mrn,
            "first_name": order.patient.first_name,
            "last_name": order.patient.last_name,
            "dob": order.patient.dob,
        },
        "provider": {
            "id": order.provider_id,
            "npi": order.provider.npi,
            "name": order.provider.name,
        },
        "order": {
            "primary_diagnosis": order.primary_diagnosis,
            "medication_name": order.medication_name,
            "additional_diagnoses": order.additional_diagnoses,
            "weight_kg": order.weight_kg,
            "allergies": order.allergies,
            "medication_history": order.medication_history,
            "patient_records": order.patient_records,
        },
    }


@dataclass
class AuditRecorder:
    audit_id: int
    started_monotonic: float

    @classmethod
    def start(
        cls,
        *,
        workflow: str,
        provider: str,
        model: str,
        rendered_prompt: str,
        input_snapshot: Mapping[str, Any],
        model_parameters: Mapping[str, Any] | None = None,
        prompt_metadata: Mapping[str, str] | None = None,
        order: Order | None = None,
        run_group_id: uuid.UUID | None = None,
        attempt_number: int = 1,
    ) -> "AuditRecorder":
        prompt_metadata = prompt_metadata or {}
        safe_input = _json_safe(input_snapshot)
        audit = GenerationAudit.objects.create(
            run_group_id=run_group_id or uuid.uuid4(),
            attempt_number=attempt_number,
            order=order,
            workflow=workflow,
            provider=provider,
            model=model,
            model_parameters=_json_safe(model_parameters or {}),
            prompt_workflow=prompt_metadata.get("prompt_workflow", ""),
            prompt_requested_version=prompt_metadata.get(
                "prompt_requested_version", ""
            ),
            prompt_version=prompt_metadata.get("prompt_version", ""),
            prompt_checksum=prompt_metadata.get("prompt_checksum", ""),
            rendered_prompt_sha256=sha256_text(rendered_prompt),
            rendered_prompt=rendered_prompt,
            input_snapshot=safe_input,
            input_sha256=sha256_text(canonical_json(safe_input)),
        )
        return cls(audit_id=audit.pk, started_monotonic=time.monotonic())

    def succeed(
        self,
        *,
        raw_output: str,
        raw_response: Any,
        token_usage: Mapping[str, Any] | None = None,
        provider_request_id: str = "",
        stop_reason: str = "",
        status: str = GenerationAudit.Status.SUCCEEDED,
        parse_success: bool | None = None,
        validation_errors: list[dict[str, Any]] | None = None,
    ) -> GenerationAudit:
        return self._finalize(
            status=status,
            raw_output=raw_output,
            raw_response=raw_response,
            token_usage=token_usage or {},
            provider_request_id=provider_request_id,
            stop_reason=stop_reason,
            parse_success=parse_success,
            validation_errors=validation_errors or [],
        )

    def fail(
        self,
        error: BaseException,
        *,
        raw_output: str = "",
        raw_response: Any = None,
        status: str = GenerationAudit.Status.FAILED,
        stop_reason: str = "",
        parse_success: bool | None = None,
        token_usage: Mapping[str, Any] | None = None,
        provider_request_id: str = "",
        validation_errors: list[dict[str, Any]] | None = None,
    ) -> GenerationAudit:
        return self._finalize(
            status=status,
            raw_output=raw_output,
            raw_response=raw_response or {},
            token_usage=token_usage or {},
            provider_request_id=provider_request_id,
            stop_reason=stop_reason,
            parse_success=parse_success,
            error_type=type(error).__name__,
            error_message=str(error),
            validation_errors=validation_errors or [],
        )

    @transaction.atomic
    def _finalize(
        self,
        *,
        status: str,
        raw_output: str,
        raw_response: Any,
        token_usage: Mapping[str, Any],
        provider_request_id: str = "",
        stop_reason: str = "",
        parse_success: bool | None = None,
        error_type: str = "",
        error_message: str = "",
        validation_errors: list[dict[str, Any]],
    ) -> GenerationAudit:
        audit = GenerationAudit.objects.select_for_update().get(
            pk=self.audit_id
        )
        if audit.completed_at is not None:
            raise ValueError("Generation audit is already finalized")

        usage = _json_safe(token_usage)
        audit.raw_output = raw_output
        audit.raw_response = _json_safe(raw_response)
        audit.output_sha256 = sha256_text(raw_output) if raw_output else ""
        audit.token_usage = usage
        audit.input_tokens = _optional_int(usage.get("input_tokens"))
        audit.output_tokens = _optional_int(usage.get("output_tokens"))
        audit.cache_creation_input_tokens = _optional_int(
            usage.get("cache_creation_input_tokens")
        )
        audit.cache_read_input_tokens = _optional_int(
            usage.get("cache_read_input_tokens")
        )
        audit.provider_request_id = provider_request_id
        audit.stop_reason = stop_reason
        audit.parse_success = parse_success
        audit.status = status
        audit.error_type = error_type
        audit.error_message = error_message
        audit.validation_errors = _json_safe(validation_errors)
        audit.completed_at = timezone.now()
        audit.duration_ms = max(
            0, round((time.monotonic() - self.started_monotonic) * 1000)
        )
        audit.record_sha256 = self._record_hash(audit)
        audit.save()
        return audit

    @staticmethod
    def _record_hash(audit: GenerationAudit) -> str:
        material = {
            "run_id": audit.run_id,
            "run_group_id": audit.run_group_id,
            "attempt_number": audit.attempt_number,
            "workflow": audit.workflow,
            "provider": audit.provider,
            "model": audit.model,
            "model_parameters": audit.model_parameters,
            "prompt_version": audit.prompt_version,
            "prompt_checksum": audit.prompt_checksum,
            "rendered_prompt_sha256": audit.rendered_prompt_sha256,
            "input_sha256": audit.input_sha256,
            "output_sha256": audit.output_sha256,
            "token_usage": audit.token_usage,
            "provider_request_id": audit.provider_request_id,
            "stop_reason": audit.stop_reason,
            "parse_success": audit.parse_success,
            "status": audit.status,
            "error_type": audit.error_type,
            "error_message": audit.error_message,
            "validation_errors": audit.validation_errors,
            "started_at": audit.started_at,
            "completed_at": audit.completed_at,
            "duration_ms": audit.duration_ms,
        }
        return sha256_text(canonical_json(material))


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (uuid.UUID, Decimal)):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
