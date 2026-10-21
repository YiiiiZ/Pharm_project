"""Minimal server-side validation for the existing intake form."""
from datetime import date
from django import forms


class OrderForm(forms.Form):
    request_key = forms.UUIDField(widget=forms.HiddenInput)
    provider_npi = forms.RegexField(r"^\d{10}$", max_length=10)
    provider_name = forms.CharField(max_length=200)
    mrn = forms.RegexField(r"^\d{6}$", max_length=6)
    first_name = forms.CharField(max_length=100)
    last_name = forms.CharField(max_length=100)
    dob = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))
    primary_diagnosis = forms.RegexField(r"^[A-Za-z][0-9][A-Za-z0-9](?:\.[A-Za-z0-9]{1,4})?$", max_length=20)
    medication_name = forms.CharField(max_length=200)
    weight_kg = forms.DecimalField(required=False, min_value=0.01, max_value=1000, decimal_places=2)
    allergies = forms.CharField(required=False, widget=forms.Textarea)
    additional_diagnoses = forms.CharField(required=False, widget=forms.Textarea)
    medication_history = forms.CharField(required=False, widget=forms.Textarea)
    patient_records = forms.CharField(required=False, widget=forms.Textarea)

    def clean_dob(self):
        dob = self.cleaned_data["dob"]
        if dob and dob > date.today():
            raise forms.ValidationError("Date of birth cannot be in the future.")
        return dob
