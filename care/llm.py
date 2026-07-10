import logging

import anthropic
from django.conf import settings

from prompts import PromptManager

from .audit import AuditRecorder, snapshot_order

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048


def generate_care_plan(order) -> str:
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    dob_str = order.patient.dob.strftime("%Y-%m-%d") if order.patient.dob else "Not provided"
    rendered_prompt = PromptManager().render(
        workflow="careplan_generation",
        scenario="web_order",
        variables={
            "patient_name": (
                f"{order.patient.first_name} {order.patient.last_name}"
            ),
            "mrn": order.patient.mrn,
            "dob": dob_str,
            "sex": "Not provided",
            "weight": (
                f"{order.weight_kg} kg"
                if order.weight_kg
                else "Not provided"
            ),
            "allergies": order.allergies or "None known",
            "provider_name": order.provider.name,
            "provider_npi": order.provider.npi,
            "medication_name": order.medication_name,
            "primary_diagnosis": order.primary_diagnosis,
            "primary_diagnosis_label": "Not provided",
            "additional_diagnoses": order.additional_diagnoses or "None",
            "medication_history": (
                order.medication_history or "None provided"
            ),
            "patient_records": order.patient_records or "None provided",
        },
    )
    logger.info(
        "Generating care plan order_id=%s prompt_metadata=%s",
        order.pk,
        rendered_prompt.metadata(),
    )

    recorder = AuditRecorder.start(
        workflow="web_careplan_generation",
        provider="anthropic",
        model=MODEL,
        rendered_prompt=rendered_prompt.content,
        input_snapshot=snapshot_order(order),
        model_parameters={
            "max_tokens": MAX_TOKENS,
            "temperature": None,
        },
        prompt_metadata=rendered_prompt.metadata(),
        order=order,
    )

    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": rendered_prompt.content}],
        )
        raw_output = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        )
        recorder.succeed(
            raw_output=raw_output,
            raw_response=message,
            token_usage=message.usage,
            provider_request_id=message.id,
            stop_reason=message.stop_reason or "",
        )
        return raw_output
    except Exception as exc:
        recorder.fail(exc)
        raise
