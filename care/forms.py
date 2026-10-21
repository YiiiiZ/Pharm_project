"""Minimal server-side validation for the existing intake form."""
from datetime import date
from django import forms


class OrderForm(forms.Form):
    request_key = forms.UUIDField(widget=forms.HiddenInput)
    provider_npi = forms.RegexField(r"^\d{10}$", max_length=10, label="Prescriber NPI", help_text="Ten-digit prescriber identifier.")
    provider_name = forms.CharField(max_length=200)
    mrn = forms.RegexField(r"^\d{6}$", max_length=6, label="Patient MRN", help_text="Six-digit patient record number.")
    first_name = forms.CharField(max_length=100)
    last_name = forms.CharField(max_length=100)
    dob = forms.DateField(label="Date of birth", required=False, widget=forms.DateInput(attrs={"type": "date"}))
    primary_diagnosis = forms.RegexField(r"^[A-Za-z][0-9][A-Za-z0-9](?:\.[A-Za-z0-9]{1,4})?$", max_length=20, help_text="ICD-10 code, for example E11.9.")
    medication_name = forms.CharField(max_length=200, help_text="Drug name only, for example metformin. Put dose and schedule in medication history.")
    weight_kg = forms.DecimalField(label="Weight (kg)", required=False, min_value=0.01, max_value=1000, decimal_places=2)
    allergies = forms.CharField(required=False, widget=forms.Textarea)
    additional_diagnoses = forms.CharField(required=False, widget=forms.Textarea)
    medication_history = forms.CharField(required=False, widget=forms.Textarea)
    patient_records = forms.CharField(required=False, widget=forms.Textarea)

    field_order = ["request_key", "mrn", "first_name", "last_name", "dob", "weight_kg", "allergies", "primary_diagnosis", "medication_name", "medication_history", "patient_records", "additional_diagnoses", "provider_name", "provider_npi"]

    def clean_dob(self):
        dob = self.cleaned_data["dob"]
        if dob and dob > date.today():
            raise forms.ValidationError("Date of birth cannot be in the future.")
        return dob
