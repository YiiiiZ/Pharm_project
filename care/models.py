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

    def __str__(self):
        return f"{self.first_name} {self.last_name} (MRN: {self.mrn})"


class Order(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    provider = models.ForeignKey(Provider, on_delete=models.CASCADE)
    primary_diagnosis = models.CharField(max_length=20)
    medication_name = models.CharField(max_length=200)
    additional_diagnoses = models.TextField(blank=True)
    medication_history = models.TextField(blank=True)
    patient_records = models.TextField(blank=True)
    care_plan = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Order #{self.pk} — {self.patient} — {self.medication_name}"
