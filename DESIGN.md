# Pharm Project — Design Document

## 1. Overview

A web-based internal tool for CVS specialty pharmacy staff (medical assistants and pharmacists) to input patient and order information, detect duplicate patients/orders, and automatically generate downloadable care plans using an LLM. Patients do not interact with this system.

**Core value proposition:** Reduce pharmacist care plan authoring time from 20–40 min to under 2 min per order.

---

## 2. Users

| Role | Primary Actions |
|---|---|
| Medical Assistant | Enter patient info, submit orders, download generated care plans |
| Pharmacist | Review care plans, export reports for pharma |

---

## 3. Data Model

### 3.1 Provider

Providers are entered once and reused across orders.

| Field | Type | Rules |
|---|---|---|
| `npi` | string (10 digits) | Primary key. Must be unique. NPI format validated (Luhn check). |
| `name` | string | Required. If NPI already exists, name must match — otherwise ERROR. |

### 3.2 Patient

| Field | Type | Rules |
|---|---|---|
| `mrn` | string (6 digits, zero-padded) | Primary key. Must be unique per patient. |
| `first_name` | string | Required |
| `last_name` | string | Required |
| `dob` | date | Required. Used for duplicate detection. |

### 3.3 Order

One order = one medication = one care plan.

| Field | Type | Rules |
|---|---|---|
| `order_id` | UUID | Auto-generated |
| `mrn` | FK → Patient | Required |
| `provider_npi` | FK → Provider | Required |
| `primary_diagnosis` | ICD-10 code | Required. Validated against ICD-10 format (`[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?`). |
| `medication_name` | string | Required |
| `additional_diagnoses` | list of ICD-10 codes | Optional. Each validated individually. |
| `medication_history` | list of strings | Optional |
| `patient_records` | text or PDF upload | Optional. Passed to LLM as context. |
| `created_at` | datetime | Auto-set |
| `care_plan` | text | LLM-generated. Stored after generation. |

---

## 4. Duplicate Detection Rules

These rules run on order submission, before writing to the database.

| Scenario | Behavior | Reason |
|---|---|---|
| Same patient (MRN) + same medication + same day | **ERROR — must block** | Definite duplicate submission |
| Same patient (MRN) + same medication + different day | **WARNING — can confirm and continue** | Possible refill / continuation |
| MRN matches, but name or DOB differs | **WARNING — can confirm and continue** | Possible data entry mistake |
| Name + DOB match, but MRN differs | **WARNING — can confirm and continue** | Possibly same person, different MRN |
| NPI matches, but provider name differs | **ERROR — must correct** | NPI is the canonical identifier; name conflict must be resolved |

All warnings must be explicitly acknowledged by the user (checkbox or confirmation dialog) before submission proceeds. All errors block submission entirely until corrected.

---

## 5. Care Plan Generation

### 5.1 Trigger
The care plan is generated automatically upon successful order submission.

### 5.2 LLM Inputs (prompt context)
- Patient demographics (name, DOB, sex, weight, allergies)
- Primary diagnosis (ICD-10 + human-readable label)
- Additional diagnoses
- Medication name
- Medication history
- Patient records (free text or extracted PDF text)

### 5.3 Required Output Sections
The LLM is instructed to produce exactly four sections:

1. **Problem List** — active clinical problems relevant to this medication order
2. **Goals** — measurable therapeutic goals for this medication
3. **Pharmacist Interventions** — specific actions the pharmacist will take
4. **Monitoring Plan** — parameters to monitor, frequency, and thresholds

### 5.4 Output Format
- Plain text, structured with labeled sections
- Downloadable as `.txt` file
- Stored in the database linked to the order

---

## 6. Web Form — Field Validation

| Field | Validation Rule |
|---|---|
| Patient First Name | Required, non-empty string, max 100 chars |
| Patient Last Name | Required, non-empty string, max 100 chars |
| Referring Provider | Required, non-empty string |
| Referring Provider NPI | Required, exactly 10 digits, passes Luhn algorithm check |
| Patient MRN | Required, exactly 6 digits (leading zeros allowed) |
| Patient DOB | Required, valid date, not in the future |
| Primary Diagnosis | Required, valid ICD-10 format |
| Medication Name | Required, non-empty string |
| Additional Diagnoses | Optional; each entry validated as ICD-10 if present |
| Medication History | Optional; each entry must be a non-empty string |
| Patient Records | Optional; text or PDF upload (max 10 MB); PDF text extracted server-side |

Validation runs both client-side (immediate feedback) and server-side (enforcement).

---

## 7. Export for Pharma Reporting

- Pharmacist can select a date range and filter by medication or provider
- Exports a CSV containing: MRN, patient name, provider NPI, medication, primary diagnosis, order date, care plan generation status
- No PHI beyond what is required for pharma reporting compliance

---

## 8. System Architecture

```
┌──────────────────────────────────────────────────┐
│                   Browser (React)                 │
│  - Web form with real-time validation             │
│  - Duplicate warning/error modals                 │
│  - Care plan viewer + download                    │
│  - Export UI                                      │
└────────────────────┬─────────────────────────────┘
                     │ HTTP/REST
┌────────────────────▼─────────────────────────────┐
│               Backend API (FastAPI)               │
│  - Input validation                               │
│  - Duplicate detection logic                      │
│  - LLM orchestration (prompt builder + caller)    │
│  - PDF text extraction                            │
│  - Export generation                              │
└──────┬─────────────────────────┬─────────────────┘
       │                         │
┌──────▼───────┐        ┌────────▼────────┐
│   Database   │        │   LLM API       │
│  (SQLite /   │        │  (Claude /      │
│  PostgreSQL) │        │   OpenAI)       │
└──────────────┘        └─────────────────┘
```

### 8.1 Module Breakdown

| Module | Responsibility |
|---|---|
| `api/routes/orders.py` | Order submission endpoint, wires validation + duplicate check + LLM |
| `api/routes/providers.py` | Provider create/lookup |
| `api/routes/patients.py` | Patient create/lookup |
| `api/routes/export.py` | CSV export endpoint |
| `core/validation.py` | All field-level validation logic |
| `core/duplicate_detection.py` | All duplicate detection rules |
| `core/care_plan.py` | Prompt construction and LLM call |
| `core/pdf_extract.py` | PDF-to-text extraction |
| `db/models.py` | SQLAlchemy models for Patient, Provider, Order |
| `db/queries.py` | DB query helpers (no raw SQL in routes) |
| `tests/` | Automated tests (see Section 9) |

---

## 9. Testing Requirements

| Test Area | What to Cover |
|---|---|
| Field validation | Each field's valid/invalid cases; boundary values (e.g. 5-digit MRN rejected) |
| NPI Luhn check | Known valid and invalid NPIs |
| ICD-10 format | Valid codes accepted; malformed codes rejected |
| Duplicate detection | All 5 scenarios from Section 4, both ERROR and WARNING paths |
| Provider dedup | Same NPI, same name: accepted. Same NPI, different name: ERROR. |
| Care plan prompt | Verify prompt contains all required fields; mock LLM response |
| PDF extraction | Valid PDF returns text; corrupt file returns safe error |
| Export | Correct columns, correct filtering by date range |

---

## 10. Error Handling

- All API errors return structured JSON: `{ "error": "...", "field": "...", "code": "..." }`
- LLM failures return a clear message; order is saved, care plan marked as `pending_retry`
- PDF parse failures do not block order submission; records field treated as empty with a warning
- No raw exceptions or stack traces exposed to the frontend

---

## 11. Out of Scope (v1)

- Patient-facing portal
- EHR/EMR integration (e.g. Epic, Cerner)
- Role-based access control beyond a single internal user role
- Multi-pharmacy / multi-tenant support
- Fax or print integration
