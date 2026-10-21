"""Single-process database queue worker; no LLM calls occur in web requests."""
from dataclasses import asdict
from datetime import timedelta
import logging
import os
from pathlib import Path

from django.utils import timezone

from rag.retrieve import retrieve_chunks, format_reference_material
from .care_plan_graph import build_care_plan_graph, make_initial_state
from .job_verification import make_evaluator
from .models import GenerationJob

logger = logging.getLogger(__name__)
CHECK_SCOPE = "Patient facts and citation IDs only. Clinical correctness and claim-to-source support require pharmacist review."


def claim_next():
    for job_id in GenerationJob.objects.filter(status="queued").order_by("created_at").values_list("pk", flat=True)[:10]:
        now = timezone.now()
        claimed = GenerationJob.objects.filter(pk=job_id, status="queued").update(
            status="running", stage="retrieval", started_at=now, updated_at=now,
        )
        if claimed:
            return GenerationJob.objects.select_related("order").get(pk=job_id)
    return None


def record_progress(job, stage):
    job.events.append({"stage": stage, "at": timezone.now().isoformat()})
    updated = GenerationJob.objects.filter(pk=job.pk, status="running").update(
        stage=stage, events=job.events, updated_at=timezone.now(),
    )
    if not updated:
        raise RuntimeError("Job is no longer running")


def fail_stale_jobs():
    """Explicit operator recovery after stopping an interrupted worker."""
    return GenerationJob.objects.filter(
        status="running", updated_at__lt=timezone.now() - timedelta(minutes=30),
    ).update(status="failed", stage="interrupted", error="Worker interrupted. You can retry this generation.", completed_at=timezone.now(), updated_at=timezone.now())


def run_job(job):
    try:
        patient = job.input_snapshot
        record_progress(job, "retrieval")
        corpus_path = os.environ.get("SPL_CHUNKS_PATH")
        if corpus_path:
            from openai import OpenAI
            from rag.embed import embed_corpus
            record_progress(job, "evidence_preparation")
            embed_corpus(input_path=Path(corpus_path), client=OpenAI(timeout=60, max_retries=1), product_name=patient["medication_name"])
            record_progress(job, "retrieval")
        # Do not send identifying patient fields to the embedding provider.
        chunks = []
        seen = set()
        for topic in ("indications and usage", "dosage and administration", "warnings precautions monitoring contraindications"):
            query = f"{patient['medication_name']} {topic}"
            for chunk in retrieve_chunks(query, product_name=patient["medication_name"], top_k=3):
                if chunk.content_sha256 not in seen:
                    seen.add(chunk.content_sha256)
                    chunks.append(chunk)
        if not chunks:
            raise ValueError("No matching drug-label evidence was found")
        job.evidence = [asdict(chunk) for chunk in chunks]
        GenerationJob.objects.filter(pk=job.pk, status="running").update(evidence=job.evidence)
        record_progress(job, "routing")
        graph = build_care_plan_graph(
            provider="anthropic", evaluator=make_evaluator(patient, job.evidence),
            audit_order=job.order, audit_run_group_id=job.pk,
        )
        state = make_initial_state(order_id=job.order_id, patient_data=patient, initial_prompt=format_reference_material(chunks))
        for event in graph.stream(state, stream_mode="updates"):
            for node, update in event.items():
                state.update(update)
                report = state.get("verification_report")
                verification = {"scope": CHECK_SCOPE, "report": asdict(report) if report else None}
                GenerationJob.objects.filter(pk=job.pk, status="running").update(
                    plan=state.get("parsed_output"), verification=verification,
                )
                next_stage = {
                    "route": "generation", "generate": "validation", "format_check": "generation" if state["status"] == "format_error" else "verification",
                    "verify": "repair" if state["status"] == "repairing" else "verification",
                    "auto_fix": "validation",
                }[node]
                record_progress(job, next_stage)
        status = "needs_review" if state["status"] in ("success", "escalated") else "failed"
        job.events.append({"stage": status, "at": timezone.now().isoformat()})
        GenerationJob.objects.filter(pk=job.pk, status="running").update(
            status=status, stage=status, events=job.events,
            error="" if status == "needs_review" else "Generation exhausted its validation or repair attempts. You can retry.",
            completed_at=timezone.now(), updated_at=timezone.now(),
        )
    except Exception as exc:
        # Persist a safe message, not a provider response or patient information.
        logger.error("Generation job %s failed (%s)", job.pk, type(exc).__name__)
        GenerationJob.objects.filter(pk=job.pk, status="running").update(
            status="failed", stage="failed", events=job.events + [{"stage": "failed", "at": timezone.now().isoformat()}], error=("No matching drug-label evidence was found. Check the medication name." if str(exc) == "No matching drug-label evidence was found" else "Generation failed. Check worker configuration and availability, then retry."),
            completed_at=timezone.now(), updated_at=timezone.now(),
        )
