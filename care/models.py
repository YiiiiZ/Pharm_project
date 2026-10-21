import uuid

from django.db import models
from django.conf import settings


class Provider(models.Model):
    npi = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=200)

    def __str__(self):
        return f"{self.name} (NPI: {self.npi})"


class Patient(models.Model):
    mrn = models.CharField(max_length=20, unique=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    dob = models.DateField(null=True, blank=True)

    def __str__(self):
        return f"{self.first_name} {self.last_name} (MRN: {self.mrn})"


class Order(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    provider = models.ForeignKey(Provider, on_delete=models.CASCADE)
    primary_diagnosis = models.CharField(max_length=20)
    medication_name = models.CharField(max_length=200)
    additional_diagnoses = models.TextField(blank=True)
    weight_kg = models.CharField(max_length=20, blank=True)
    allergies = models.TextField(blank=True)
    medication_history = models.TextField(blank=True)
    patient_records = models.TextField(blank=True)
    care_plan = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Order #{self.pk} — {self.patient} — {self.medication_name}"


class GenerationAudit(models.Model):
    class Status(models.TextChoices):
        STARTED = "started", "Started"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"
        REFUSED = "refused", "Refused"
        TRUNCATED = "truncated", "Truncated"
        PARSE_FAILED = "parse_failed", "Parse failed"

    run_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    run_group_id = models.UUIDField(default=uuid.uuid4, db_index=True)
    attempt_number = models.PositiveSmallIntegerField(default=1)
    order = models.ForeignKey(
        Order,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="generation_audits",
    )

    workflow = models.CharField(max_length=100)
    provider = models.CharField(max_length=50)
    model = models.CharField(max_length=200)
    model_parameters = models.JSONField(default=dict)

    prompt_workflow = models.CharField(max_length=100, blank=True)
    prompt_requested_version = models.CharField(max_length=50, blank=True)
    prompt_version = models.CharField(max_length=50, blank=True)
    prompt_checksum = models.CharField(max_length=64, blank=True)
    rendered_prompt_sha256 = models.CharField(max_length=64, blank=True)
    rendered_prompt = models.TextField()

    input_snapshot = models.JSONField(default=dict)
    input_sha256 = models.CharField(max_length=64)
    raw_output = models.TextField(blank=True)
    raw_response = models.JSONField(default=dict)
    output_sha256 = models.CharField(max_length=64, blank=True)

    input_tokens = models.PositiveIntegerField(null=True, blank=True)
    output_tokens = models.PositiveIntegerField(null=True, blank=True)
    cache_creation_input_tokens = models.PositiveIntegerField(
        null=True, blank=True
    )
    cache_read_input_tokens = models.PositiveIntegerField(null=True, blank=True)
    token_usage = models.JSONField(default=dict)

    provider_request_id = models.CharField(max_length=200, blank=True)
    stop_reason = models.CharField(max_length=100, blank=True)
    parse_success = models.BooleanField(null=True, blank=True)
    status = models.CharField(
        max_length=30,
        choices=Status.choices,
        default=Status.STARTED,
        db_index=True,
    )
    error_type = models.CharField(max_length=200, blank=True)
    error_message = models.TextField(blank=True)
    validation_errors = models.JSONField(default=list)

    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_ms = models.PositiveIntegerField(null=True, blank=True)
    record_sha256 = models.CharField(max_length=64, blank=True)

    class Meta:
        ordering = ["-started_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["run_group_id", "attempt_number"],
                name="unique_generation_audit_attempt",
            )
        ]

    def save(self, *args, **kwargs):
        if self.pk:
            original = type(self).objects.filter(pk=self.pk).only(
                "completed_at"
            ).first()
            if original and original.completed_at is not None:
                raise ValueError("Finalized generation audits are immutable")
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise ValueError("Generation audit records cannot be deleted")

    def __str__(self):
        return (
            f"{self.workflow} {self.provider}/{self.model} "
            f"{self.run_id} [{self.status}]"
        )


class GenerationJob(models.Model):
    """A durable queue entry and immutable generation result for one order."""

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        NEEDS_REVIEW = "needs_review", "Ready for pharmacist review"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    request_key = models.UUIDField(unique=True)
    order = models.ForeignKey(Order, on_delete=models.PROTECT, related_name="jobs")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.QUEUED)
    stage = models.CharField(max_length=40, default="queued")
    input_snapshot = models.JSONField(default=dict)
    evidence = models.JSONField(default=list)
    plan = models.JSONField(null=True, blank=True)
    verification = models.JSONField(default=dict)
    events = models.JSONField(default=list)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["order"],
                condition=models.Q(status__in=["queued", "running"]),
                name="one_active_generation_per_order",
            ),
        ]
        indexes = [models.Index(fields=["status", "created_at"])]


class ReviewDecision(models.Model):
    class Decision(models.TextChoices):
        APPROVED = "approved", "Approved"
        REJECTED = "rejected", "Rejected"
        CHANGES_REQUESTED = "changes_requested", "Changes requested"

    job = models.OneToOneField(GenerationJob, on_delete=models.PROTECT, related_name="review")
    reviewer = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT)
    reviewer_name = models.CharField(max_length=200)
    decision = models.CharField(max_length=20, choices=Decision.choices)
    comments = models.TextField(blank=True)
    plan_sha256 = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        permissions = [("review_careplan", "Can approve or reject care plans")]

    def save(self, *args, **kwargs):
        if not self._state.adding:
            raise ValueError("Review decisions cannot be changed; review a new generation instead.")
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise ValueError("Review decisions cannot be deleted.")
