import uuid

from django.db import models


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
