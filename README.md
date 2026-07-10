# Specialty Pharmacy Care Plan Generator

An internal Django prototype for generating pharmacist-reviewed specialty
pharmacy care plans from patient records and drug-label evidence.

The repository contains three layers:

- A working Django MVP that submits orders to Claude, stores free-text care
  plans in SQLite, and supports viewing, printing, and text download.
- A newer RAG and structured-generation foundation covering FDA label
  ingestion, versioned prompts, typed care-plan JSON, OpenAI/Claude structured
  output, validation, and bounded repair retries.
- A pharmacist-facing query interface built with OpenAI Agents SDK that routes
  natural-language queries to specialist agents for progress checks,
  verification report details, and care-plan regeneration requests.

The generation pipeline and query interface are implemented and tested as
reusable Python modules, but are not yet connected to the Django order form.
This project is not ready for production or independent clinical use. Every
generated plan requires qualified pharmacist review.

## Current progress

### Working web MVP

- Browser-based order intake for provider, patient, medication, diagnosis, allergies, medication history, and clinical notes
- Reuse of providers by NPI and patients by MRN
- Synchronous free-text care-plan generation with `claude-sonnet-4-6`
- Prompt-enforced Problem List, Goals, Pharmacist Interventions, and Monitoring Plan sections
- SQLite persistence for providers, patients, orders, and generated plans
- Care-plan review, browser printing, and plain-text download
- Docker and Docker Compose setup

### RAG knowledge-base preparation

- FDA SPL ZIP/XML ingestion without extracting archive contents
- Clinical-section extraction with product, version, NDC, ingredient, route, and source metadata
- Removal of XML, packaging, display-panel, image-reference, and manufacturer boilerplate noise
- Section/subsection-aware chunking with lists and tables preserved where possible
- Default chunking policy of 550 target tokens, 850 maximum tokens, and 75-token overlap
- Full May 2026 prescription-label corpus processed:
  - 1,842 archives
  - 74,952 clinical sections
  - 91,357 chunks
  - Zero parse failures
- JSONL output ready for embedding and vector-database ingestion

### Generation quality infrastructure

- Complexity router that selects prompt version and model from patient-record complexity
- Orchestrator that combines routing and hallucination-repair loop execution
- Versioned prompts selected by workflow and scenario through `config.yaml`
- Prompt version, resolved version, and SHA-256 checksum metadata
- Pydantic care-plan schema for `patient_summary`, `problems`, `goals`, and `interventions`
- Required-field validation, controlled type coercion, exact JSON-path errors, and rejection of unexpected fields
- Evaluator-generator retry loop that feeds hallucination feedback back into the next prompt
- Provider-native structured output:
  - OpenAI Responses API with `text_format=CarePlan`
  - Claude Messages API with `output_format=CarePlan`
- Validation-feedback retry loop with at most two repair attempts
- `parse_failed` result preserving original output, final output, all attempts, and validation errors
- Automatic LLM generation audit records for the live Claude path and optional structured-provider runs
- Audit fields for order ID, prompt version/checksum, model parameters, token usage, duration, full input/output, provider request ID, status, and parse outcome
- Retry attempts grouped under a shared run-group ID
- Read-only order audit endpoint: `GET /api/llm-logs?order_id=<id>`
- Versioned prompts for free-text generation, structured generation, review, and repair
- Experimental deterministic checks for weight, ICD-10, and potentially ungrounded medication terms
- Three parallel hallucination-verification strategies (Racing, Voting, Sectioning) implemented as drop-in `Evaluator` replacements
- `risk_level` classification on individual findings (`low` / `medium` / `high`) and pharmacist escalation flag on reports
- LangGraph multi-step orchestration pipeline with five nodes, two conditional edges, and full per-step state tracking
- Typed shared state (`CarePlanState`) covering all retry counters, prompt version, model, LLM output, verification report, error history, and escalation flag
- Accumulated feedback prompt built from the full error log and injected into every subsequent generation attempt
- Deterministic auto-fix node that patches numerical hallucinations from ground-truth patient data without an extra LLM call
- Three end-to-end scenario tests: one-time pass, parse-failure prompt-version retry, hallucination-detected regeneration

### Pharmacist query interface

A conversational query layer built with OpenAI Agents SDK (`care/query_agents.py`)
allows pharmacists to ask natural-language questions about care plans in progress.

TriageAgent classifies intent and hands off to the correct specialist without
answering questions itself. Each specialist agent carries only the tools it needs:

| Agent | Responsibility | Tools |
|---|---|---|
| TriageAgent | Intent classification and patient ID extraction; routes to one of the three specialists | None |
| ProgressAgent | Pipeline stage and timing queries | `get_care_plan_status`, `get_status_timeline`, `get_estimated_completion` |
| ReviewAgent | Verification report details and flagged issues | `get_verification_issues`, `get_verification_report`, `get_sub_report` |
| ResubmitAgent | Eligibility check and async regeneration submission | `check_regeneration_eligibility`, `submit_regeneration`, `get_regeneration_status` |

ResubmitAgent always calls `check_regeneration_eligibility` before
`submit_regeneration`, returns a `job_id` immediately (the LangGraph workflow
runs asynchronously), and will not submit more than once per turn.

Specialist agents hand back to TriageAgent when the pharmacist switches topic,
keeping cross-topic routing centralised.

All tools are currently mocked and return representative data. Replace the mock
bodies with live API calls to connect to the LangGraph pipeline and persistence
layer.

### Evaluation and tests

- Evaluation dataset covering 10 medication scenarios
- Keyword-based evaluation script
- Frozen five-case prompt-version benchmark covering IVIG, vancomycin, methotrexate, omalizumab, and warfarin
- Automatic prompt-run comparison for JSON parse success, accuracy proxy, coverage, F1, and token usage
- Optional pricing-based estimated API cost and regression gates
- Current development checkpoint: macro F1 0.86 and micro F1 0.86
- Over 80 automated tests covering prompt management, schema validation, structured-provider adapters, retry behavior, audit logging/API, RAG chunking, prompt benchmarking, routing, the evaluator-generator loop, orchestration, parallel verification, and LangGraph scenario execution

### Remaining work

- Server-side clinical field validation, including NPI and ICD-10 validation
- Duplicate-order warnings and blocking rules described in `DESIGN.md`
- Provider/patient conflict handling when an existing NPI or MRN is submitted with different demographics
- Authentication, authorization, audit-access logging, and production security controls
- PDF upload and text extraction
- CSV reporting/export
- Embedding generation and vector-database storage
- Runtime semantic/hybrid retrieval and reranking
- Injection of retrieved label chunks into the live Django generation request
- Migration of the web UI from free text to structured care-plan JSON
- Persistence of retrieval results, citations, and verification results
- Authentication and authorization for the audit-log API
- Retention, archival, and redaction policy for audit records containing PHI
- Integration of deterministic and claim-to-source verification into the web workflow
- Background jobs, transport-level retries, and user-facing failure handling
- End-to-end Django request tests and clinical retrieval/grounding evaluations

See [DESIGN.md](DESIGN.md) for the intended product scope. Some design choices in that document describe the target system rather than the current Django MVP.

## Tech stack

- Python 3.11 in Docker
- Django 5.0
- SQLite
- Anthropic and OpenAI Python SDKs
- Pydantic 2
- LangGraph 1.x (multi-step orchestration pipeline)
- OpenAI Agents SDK 0.17.x (pharmacist query interface)
- PyYAML
- Bootstrap 5 via CDN
- JSONL as the current RAG interchange format
- Vector database not yet selected

## Run with Docker

### Prerequisites

- Docker with Docker Compose
- An Anthropic API key for the current web MVP

### Setup

```bash
cp .env.example .env
```

Set `ANTHROPIC_API_KEY` and replace the example Django secret in `.env`, then
run:

```bash
docker compose up --build
```

Open <http://localhost:8000>.

The SQLite database is stored in the Docker volume `db_data`.

The current Compose service passes only the variables required by the live
Claude web path. To run the OpenAI structured adapter inside Docker, add
`OPENAI_API_KEY` and `OPENAI_MODEL` to the service environment first.

## Run locally

Python 3.11 is recommended to match the Docker image.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set the values in `.env`, export them into the shell, and initialize the database:

```bash
set -a
source .env
set +a
mkdir -p data
python manage.py migrate --run-syncdb
python manage.py runserver
```

Open <http://127.0.0.1:8000>.

## Current web application flow

1. Staff enter provider, patient, order, and clinical information.
2. Django reuses or creates the provider by NPI and patient by MRN.
3. A new order is saved.
4. The configured prompt version is loaded and rendered by `PromptManager`.
5. The application sends the rendered prompt to Claude and waits for a response.
6. Claude returns a free-text care plan.
7. The generated care plan is stored and displayed.
8. Staff can print it or download it as a `.txt` file.

Generation currently happens inside the web request and can take 15–30 seconds.

## Current generation pipeline

The live Django web path generates a free-text care plan synchronously and
stores it against the order:

```text
Input
  -> Generate Care Plan (Claude)
  -> Layer 1 Verification (code-based comparison)
  -> Layer 2 Review (LLM-as-Judge)
  -> Store in database
  -> Pharmacist review
```

The newer LangGraph pipeline (not yet wired to the Django form) adds
complexity routing, prompt-version retries, parallel hallucination
verification, and deterministic auto-correction:

```text
Input
  -> route (complexity router: simple/complex → prompt version + model)
  -> generate (schema-constrained LLM call)
  -> format_check (pass-through for structured output; validates raw text)
         ├── parse ok            → verify
         ├── parse failed, retry → generate (next prompt version, with error history)
         └── retries exhausted   → END (failed)
  -> verify (parallel hallucination evaluator: Racing / Voting / Sectioning)
         ├── clean               → END (success)
         ├── high-risk finding   → END (escalated to pharmacist)
         ├── numerical finding   → auto_fix (patch weight/ICD-10 from patient data)
         └── reasoning finding   → generate (regenerate with accumulated feedback)
```

Source attribution is included in the generation context, and the audit log
records the prompt version, model, token usage, full input/output, and parse
status for each run.

## Target RAG flow

The implemented modules support the following target flow, but the steps are
not yet connected end to end:

```text
Patient record and medication
        ↓
Semantic/hybrid retrieval from SPL chunks       not yet implemented
        ↓
Patient record + retrieved reference chunks
        ↓
Versioned structured-generation prompt          implemented
        ↓
OpenAI or Claude schema-constrained output       implemented
        ↓
Pydantic and post-generation validation          implemented
        ↓
Validation-feedback repair, maximum 2 retries    implemented
        ↓
Grounding checks and pharmacist approval         partial/manual
```

## LLM generation audit log

Every live Django care-plan API call automatically creates a
`GenerationAudit` row before contacting Claude. The row is finalized on
success or failure and then becomes application-level immutable.

Recorded fields include:

- `order_id`, unique `run_id`, shared retry `run_group_id`, and attempt number
- Workflow, provider, model, and complete model parameters
- Prompt workflow, requested version, resolved version, and template checksum
- Fully rendered prompt and its SHA-256 checksum
- Immutable patient/order input snapshot and checksum
- Complete raw model output and complete provider response
- Input, output, cache-creation, and cache-read token usage
- Start/completion timestamps and duration in milliseconds
- Provider request ID and stop reason
- `parse_success`, final status, error details, and validation errors
- Final record checksum for later integrity checking

The current free-text web path records `parse_success=null` because no JSON
parse is attempted. Structured OpenAI/Claude calls record `true` or `false`
when auditing is enabled:

```python
plan = generate_structured_care_plan(
    provider="anthropic",
    patient_record=patient_record,
    reference_material=retrieved_chunks,
    audit_enabled=True,
    audit_order=order,
)
```

Structured retry calls can audit each attempt under one run group:

```python
result = generate_structured_care_plan_with_retry(
    provider="anthropic",
    patient_record=patient_record,
    reference_material=retrieved_chunks,
    audit_enabled=True,
    audit_order=order,
    max_retries=2,
)
```

Query all runs for an order:

```http
GET /api/llm-logs?order_id=123
```

Optional pagination:

```http
GET /api/llm-logs?order_id=123&limit=20&offset=0
```

The response includes the complete snapshots and raw input/output:

```json
{
  "order_id": 123,
  "count": 1,
  "limit": 50,
  "offset": 0,
  "results": [
    {
      "run_id": "...",
      "prompt": {"version": "v3", "rendered": "..."},
      "model": "claude-sonnet-4-6",
      "token_usage": {"input_tokens": 100, "output_tokens": 25},
      "duration_ms": 842,
      "parse_success": null,
      "status": "succeeded",
      "input_snapshot": {},
      "raw_output": "...",
      "raw_response": {}
    }
  ]
}
```

Initialize the table with the project’s current migration strategy:

```bash
python manage.py migrate --run-syncdb
```

The endpoint currently has no authentication because the MVP has no auth
layer. It exposes full prompts, patient snapshots, and outputs, so it must not
be exposed outside a controlled development environment until access control,
audit access logging, encryption, and retention policies are implemented.

## Prompt versioning

Prompts are stored outside application code:

```text
prompts/
  manager.py
  config.yaml
  careplan_generation/
    v1.txt
    v2.txt
    v3.txt
    current.txt
  careplan_review/
    v1.txt
    current.txt
  careplan_structured/
    v1.txt
    current.txt
  careplan_repair/
    v1.txt
    current.txt
```

`config.yaml` defines available versions, `current` aliases, defaults, and
scenario-specific selections. `current.txt` is a checked-in copy of the
active prompt so deployments do not depend on symlink behavior.

Templates use Python `string.Template` variables such as `$patient_name`.
Rendering is strict: a missing variable raises `PromptRenderError`.

```python
from prompts import PromptManager

rendered = PromptManager().render(
    workflow="careplan_review",
    scenario="evaluation",
    variables={
        "source_record": source_record,
        "care_plan": care_plan,
    },
)

print(rendered.content)
print(rendered.metadata())
```

Metadata includes the workflow, requested version, concrete resolved version,
and a short SHA-256 checksum. Evaluation runs persist these fields in
`eval/dataset.json`; web generation writes them to application logs.

When promoting a new prompt:

1. Add a new immutable version file such as `v4.txt`.
2. Register it under `versions` in `prompts/config.yaml`.
3. Point the relevant scenario and `current` alias to `v4`.
4. Copy `v4.txt` to `current.txt`.
5. Run the prompt manager tests and evaluation suite before release.

Current aliases:

| Workflow | Current version | Purpose |
|---|---|---|
| `careplan_generation` | `v3` | Free-text web/evaluation generation |
| `careplan_review` | `v1` | LLM-as-Judge review prompt |
| `careplan_structured` | `v2` | RAG-grounded structured generation |
| `careplan_repair` | `v1` | Validation-error repair attempts |

## Evaluation

`eval/dataset.json` contains 10 clinical scenarios:

- IVIG
- Vancomycin
- Enoxaparin
- Methotrexate
- Infliximab
- Rituximab
- Somatropin
- Omalizumab
- Palivizumab
- Warfarin

Run every case through the configured model:

```bash
python eval/run_eval.py
```

This makes paid Anthropic API calls and appends results to `eval/dataset.json`.

Score the latest run for each case:

```bash
python eval/score_eval.py
```

At the current checkpoint, the latest runs produce:

| Aggregate | Precision | Recall | F1 |
|---|---:|---:|---:|
| Macro average | 0.80 | 0.93 | 0.86 |
| Micro average | 0.80 | 0.92 | 0.86 |

These are development metrics, not evidence of clinical safety or efficacy. Coverage is determined by keyword matching, and false-positive counts are manually estimated in `eval/score_eval.py`.

## Prompt-version benchmark

Before promoting a structured care-plan prompt, run the fixed benchmark in
`eval/prompt_benchmark.yaml`. It selects five clinically diverse fictional
cases from `eval/dataset.json`:

| Case | Medication | Main challenge |
|---|---|---|
| `case_001` | IVIG | Infusion, renal, thrombosis, and respiratory risks |
| `case_002` | Vancomycin | Renal dosing and therapeutic monitoring |
| `case_004` | Methotrexate | Weekly-dose safety and laboratory monitoring |
| `case_008` | Omalizumab | Biologic dosing and anaphylaxis monitoring |
| `case_010` | Warfarin | Major interaction and INR management |

The case IDs are fixed and every run stores a SHA-256 hash of the selected
inputs and expected criteria. Runs with different hashes are not compared.
Reference evidence is also held constant, so this benchmark measures prompt
and generation performance rather than retrieval quality.

Metrics:

- **JSON parse success rate:** valid `CarePlan` responses / attempted cases
- **Accuracy:** golden-precision proxy; generated clinical items matching at
  least one expected criterion / all generated clinical items
- **Coverage:** expected criteria found anywhere in the generated plan / all
  expected criteria
- **F1:** harmonic mean of the accuracy proxy and coverage
- **Token cost:** input/output token totals and averages
- **Estimated dollar cost:** only calculated when prices are explicitly
  supplied; pricing is not hardcoded

Run a prompt version:

```bash
python -m eval.prompt_benchmark \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --prompt-version v1
```

The runner saves a detailed JSON report under the ignored directory
`eval/prompt_benchmark_results/`. It automatically compares the result with
the most recent run using the same provider, model, and test-set hash.

Compare against an explicit baseline and fail the command on configured
regressions:

```bash
python -m eval.prompt_benchmark \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --prompt-version v2 \
  --baseline eval/prompt_benchmark_results/<v1-result>.json \
  --fail-on-regression
```

Optional cost estimate:

```bash
python -m eval.prompt_benchmark \
  --provider openai \
  --prompt-version v2 \
  --input-price-per-million <current-input-price> \
  --output-price-per-million <current-output-price>
```

Default comparison gates are defined in `eval/prompt_benchmark.yaml`:

- No parse-success regression
- At most 5 percentage points of accuracy regression
- At most 5 percentage points of coverage regression
- At most 20% increase in total tokens

The accuracy and coverage metrics use deterministic keyword/concept matching.
They are suitable for relative prompt comparison, not clinical validation.
Review per-item misses and unexpected generated items before promoting a
prompt even when the aggregate gate passes.

Current structured-prompt leaderboard:

| Version | Parse success | Accuracy | Coverage | F1 | Total tokens | Status |
|---|---:|---:|---:|---:|---:|---|
| `v1` | 100.0% | 97.9% | 99.0% | 98.4% | 25,680 | Baseline |
| `v2` | 100.0% | 97.8% | 99.0% | 98.4% | 22,793 | Current best |
| `v3` | 100.0% | 97.9% | 98.0% | 98.0% | 23,793 | Rejected |

`v2` is current because it retained the baseline’s parse success, coverage,
and rounded F1 while reducing total token use by 11.2%. See
`prompts/careplan_structured/BENCHMARK.md` for the recorded decision.

## Preparing SPL labels for RAG

`rag/prepare_spl.py` streams FDA Structured Product Labeling XML directly
from ZIP archives, extracts product metadata and clinical sections, removes
display/package noise, and writes section-aware chunks as JSON Lines.

The default chunking policy is:

```yaml
target_tokens: 550
max_tokens: 850
overlap_tokens: 75
```

Structural boundaries take priority over size: separate label sections are
never combined, nested subsection ancestry is retained, and paragraphs,
lists, and table rows are kept together when possible. Token counts currently
use a deterministic regex estimate; use the selected embedding model's
tokenizer if exact billing or context limits are required.

Prepare the human prescription labels:

```bash
python -m rag.prepare_spl \
  /path/to/dm_spl_monthly_update_may2026 \
  --category prescription \
  --output data/rag/spl_chunks.jsonl
```

Test against a small sample first:

```bash
python -m rag.prepare_spl \
  /path/to/dm_spl_monthly_update_may2026 \
  --category prescription \
  --output /tmp/spl_sample.jsonl \
  --limit 10
```

Each JSONL record contains:

- Stable `chunk_id`
- Clean text with drug and qualified section headings
- Estimated token count and content checksum
- SPL set/document IDs and label version
- Product and generic names
- NDCs, route, dosage form, and active ingredients
- Section code, title, parent section, and effective date
- Source archive and XML filename for citation traceability

The derived `data/` output is ignored by Git. Embed `text`, use `chunk_id` as
the vector point ID, and store `metadata` plus `content_sha256` as the vector
database payload. Filter retrieval by product/generic name, document type,
label status/version, and section when appropriate. Deduplicate final search
results by `content_sha256` because repackaged labels can contain identical
clinical text.

The generated May 2026 prescription corpus currently exists locally at:

```text
data/rag/spl_chunks.jsonl
data/rag/spl_chunks.jsonl.manifest.json
```

It is derived data and is intentionally not committed.

## Experimental care-plan verification

`care/verify_care_plan.py` performs deterministic checks against a stored order:

- Weight mentions match the recorded weight
- The primary ICD-10 code appears in the generated plan
- Extracted medication-like terms are grounded in selected order fields

Run it for an existing order:

```bash
python care/verify_care_plan.py <order_id>
```

The medication-term extraction is heuristic and may produce false positives. This verifier is not currently called during normal web requests.

## Parallel hallucination verification

`care/parallel_verifier.py` provides three verification strategies that all
conform to the `Evaluator` protocol (`(output: str) -> HallucinationReport`)
and can therefore replace a single evaluator in any retry loop or LangGraph
node without changing the surrounding code.

### RacingVerifier

Submits the same output to multiple evaluators concurrently and returns the
most thorough result. The default `select="most_findings"` keeps the report
with the highest hallucination count — the conservative choice for a clinical
tool. `select="fastest"` returns the first result that completes.

```python
from care.parallel_verifier import RacingVerifier

verifier = RacingVerifier(
    [claude_evaluator, gpt_evaluator],
    select="most_findings",
)
report = verifier(care_plan_text)
```

### VotingVerifier

Runs all evaluators in parallel and decides validity by strict majority vote.
On a tie the result is invalid. Findings from all evaluators are merged and
deduplicated by path.

```python
from care.parallel_verifier import VotingVerifier

verifier = VotingVerifier([evaluator_a, evaluator_b, evaluator_c])
report = verifier(care_plan_text)
```

### SectioningVerifier

Splits the care plan into its four sections (Problem List, Goals,
Pharmacist Interventions, Monitoring Plan) and evaluates each section in its
own thread with a dedicated evaluator. Merge rules are conservative:

- Any section invalid → overall invalid
- Any finding with `risk_level="high"` → `escalate_to_pharmacist=True`
- Each finding's path is prefixed with its section name for traceability

This is the primary strategy for catching reasoning errors. Each evaluator is
scoped to one clinical domain, reducing cross-domain hallucination and making
individual failures easier to locate.

```python
from care.parallel_verifier import SectioningVerifier

verifier = SectioningVerifier(
    {
        "problems":      problems_evaluator,
        "goals":         goals_evaluator,
        "interventions": interventions_evaluator,
        "monitoring":    monitoring_evaluator,
    }
)
report = verifier(care_plan_text)

if report.escalate_to_pharmacist:
    # route to human review
    ...
```

### Finding risk levels

`HallucinationFinding` carries a `risk_level` field (`"low"` / `"medium"` /
`"high"`). The LangGraph pipeline uses this to route between auto-correction,
model regeneration, and pharmacist escalation:

| risk_level | Action |
|---|---|
| `high` | Escalate to pharmacist unconditionally |
| `medium` | Regenerate with full error context in prompt |
| `low` | Regenerate with full error context in prompt |
| numerical path | Patch from patient data without another LLM call |

## LangGraph orchestration pipeline

The LangGraph pipeline (`care/care_plan_graph.py`) replaces the flat
`CarePlanOrchestrator` for workflows requiring multi-step branching: prompt
version retries, finding-type-dependent repair strategies, and future
human-in-the-loop checkpoints.

### State

`CarePlanState` (`care/graph_state.py`) is a single `TypedDict` shared by
every node. All retry counters, prompt version, model, outputs, verification
results, and error history live here. Nodes return only the keys they changed;
LangGraph merges updates automatically.

Key fields:

| Field | Written by | Purpose |
|---|---|---|
| `routing_strategy` | `route` | Selected model and prompt version |
| `current_prompt_version` | `route`, `format_check` | Advances on format failure |
| `llm_output` / `parsed_output` | `generate` | Raw and parsed care plan |
| `verification_report` | `verify` | Hallucination findings and counts |
| `format_retry_count` | `generate`, `format_check` | Prompt-version retries used |
| `repair_retry_count` | `verify` | Hallucination-repair retries used |
| `error_log` | any failing node | Typed records of every failure |
| `accumulated_feedback` | any failing node | Full error history as prompt suffix |
| `escalate_to_pharmacist` | `verify` | Set on any high-risk finding |
| `status` | every node | Current `WorkflowStatus` value |

### Nodes

| Node | Source | Responsibility |
|---|---|---|
| `route` | `route_node` | `assess_complexity` → sets prompt version and model |
| `generate` | `make_generate_node(provider, client)` | One schema-constrained LLM call; logs format errors and advances prompt version |
| `format_check` | `format_check_node` | Pass-through for structured output; validates raw text and advances version on failure |
| `verify` | `make_verify_node(evaluator)` | Runs the supplied `Evaluator`; classifies findings and sets repair strategy |
| `auto_fix` | `auto_fix_node` | Patches numerical fields (`weight_kg`, `primary_diagnosis`, `mrn`) from `patient_data` without an LLM call |

`generate` and `verify` are created by factories so the LLM client and
evaluator are injected at construction time. Nodes are testable as plain
functions without running the graph.

### Edges

`care/graph_edges.py` defines the two conditional routing functions:

**`route_after_format`** — after `format_check`:

| State | Next node |
|---|---|
| `status=VERIFYING` | `verify` |
| `status=FORMAT_ERROR`, retries remaining | `generate` |
| `status=FORMAT_ERROR`, retries exhausted | `END` (failed) |

**`route_after_verify`** — after `verify`:

| State | Next node |
|---|---|
| `status=SUCCESS` | `END` |
| `status=ESCALATED` | `END` |
| `REPAIRING`, all findings numerical | `auto_fix` |
| `REPAIRING`, any reasoning finding, retries remaining | `generate` |
| `REPAIRING`, retries exhausted | `END` (failed) |

A finding is classified as numerical when its path references a factual
patient field (`weight`, `primary_diagnosis`, `mrn`, `lab`). Everything in
`problems`, `goals`, `interventions`, or `monitoring` is a reasoning finding.

### Building and running the graph

```python
from care.care_plan_graph import build_care_plan_graph, make_initial_state
from care.parallel_verifier import SectioningVerifier

compiled = build_care_plan_graph(
    provider="anthropic",
    evaluator=SectioningVerifier(
        section_evaluators={...},
        default_evaluator=fallback_evaluator,
    ),
)

state = make_initial_state(
    order_id=42,
    patient_data=patient_dict,
    initial_prompt="Generate a structured care plan for this order.",
)

result = compiled.invoke(state)

if result["status"] == "success":
    care_plan = result["final_output"]
elif result["status"] == "escalated":
    # route to pharmacist queue
    ...
```

Stream state changes step by step:

```python
for event in compiled.stream(state, stream_mode="updates"):
    for node_name, update in event.items():
        print(node_name, update)
```

Generate the pipeline diagram:

```python
from care.care_plan_graph import save_graph_diagram

save_graph_diagram("care_plan_graph.md")   # Mermaid source
# Optional PNG via Mermaid.ink (requires internet):
compiled.get_graph().draw_mermaid_png()
```

### Scenario tests

`care/test_graph_scenarios.py` runs three mocked end-to-end scenarios that
require no API keys:

```bash
python care/test_graph_scenarios.py
```

Or as pytest:

```bash
python -m pytest care/test_graph_scenarios.py -v
```

| Scenario | What is tested |
|---|---|
| One-time pass | Clean generation and verification in four steps |
| Parse failure → retry | `StructuredOutputError` twice; prompt advances v2 → v3; succeeds on third attempt; `accumulated_feedback` contains both error messages |
| Hallucination → regen | Fabricated guideline reference flagged as medium/reasoning; regeneration carries the finding detail; second attempt is clean |

## Care-plan JSON schema validation

`care/care_plan_schema.py` defines the expected structured output with
Pydantic. The four required top-level fields are:

- `patient_summary`
- `problems`
- `goals`
- `interventions`

Validate raw LLM output before saving or using it:

```python
from care.care_plan_schema import validate_care_plan

result = validate_care_plan(llm_response_text)

if result.valid:
    normalized_care_plan = result.data
else:
    for error in result.errors:
        print(error.path, error.message, error.error_type)
```

Pydantic performs controlled coercion, including converting `"72"` to
`72.0`. The project also accepts a single allergy string as a one-item list
and common string booleans such as `"yes"` and `"false"`.

Example validation error:

```json
{
  "path": "$.goals[0].target",
  "message": "Field required",
  "error_type": "missing",
  "input": {
    "description": "Preserve renal function"
  }
}
```

Unknown fields are rejected so an unexpected LLM output shape does not pass
silently. The same model can produce JSON Schema for inclusion in a prompt or
for an API that supports constrained structured output:

```python
from care.care_plan_schema import care_plan_json_schema

schema = care_plan_json_schema()
```

## Provider-native structured output

`care/structured_llm.py` sends the `CarePlan` Pydantic model directly to
OpenAI or Claude so generation is constrained to the schema at decoding time.
Both adapters return the same validated `CarePlan` object.

```python
from care.structured_llm import generate_structured_care_plan

plan = generate_structured_care_plan(
    provider="anthropic",  # or "openai"
    patient_record=patient_record,
    reference_material=retrieved_chunks,
)

care_plan_json = plan.model_dump(mode="json")
```

Provider-specific methods:

```python
from care.structured_llm import (
    generate_anthropic_care_plan,
    generate_openai_care_plan,
)
```

- OpenAI uses `client.responses.parse(..., text_format=CarePlan)`.
- Claude uses `client.messages.parse(..., output_format=CarePlan)`.
- Refusals, truncated responses, and missing parsed output raise
  `StructuredOutputError`.
- The RAG grounding prompt is versioned under
  `prompts/careplan_structured/`.

Configure the provider through environment variables:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.5
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-4-6
```

Structured output guarantees the JSON shape, not clinical correctness or
source grounding. Continue running deterministic checks, claim-to-source
verification, and pharmacist review.

## Validation-feedback retries

`care/care_plan_retry.py` implements a separate retry policy for invalid model
content:

```text
generate → validate
              ├── valid → success
              └── invalid → send exact errors + prior output back to model
                                      ↓
                              retry up to 2 times
                                      ↓
                          parse_failed if still invalid
```

Using the provider adapters:

```python
from care.care_plan_retry import (
    generate_structured_care_plan_with_retry,
)

result = generate_structured_care_plan_with_retry(
    provider="anthropic",
    patient_record=patient_record,
    reference_material=retrieved_chunks,
    max_retries=2,
)

if result.status == "success":
    care_plan = result.data
else:
    # Persist this result for debugging/manual review.
    failed_record = result.model_dump(mode="json")
```

`max_retries=2` permits three total generations. Each failed attempt stores:

- Raw output
- Exact field-path validation errors
- Attempt number

If all attempts fail, the result has `status="parse_failed"` and preserves
`original_output`, `final_output`, and the complete `attempts` history.

For a raw API generator or test double:

```python
from care.care_plan_retry import generate_with_validation_retry

def generator(*, attempt_number, repair_instructions):
    return call_llm(
        base_prompt=base_prompt,
        repair_instructions=repair_instructions,
    )

result = generate_with_validation_retry(generator, max_retries=2)
```

The retry prompt is versioned under `prompts/careplan_repair/`. API timeouts,
rate limits, and network failures are intentionally not marked
`parse_failed`; handle those separately with exponential backoff because no
model output may exist to preserve.

## Project structure

```text
care/
  audit.py                   Generation audit recorder and order snapshots
  care_plan_schema.py        Pydantic care-plan JSON schema and validation
  care_plan_retry.py         Validation feedback and bounded repair retries
  structured_llm.py          OpenAI and Claude structured-output adapters
  llm.py                     Current free-text Claude generation path
  models.py                  Provider, Patient, and Order models
  views.py                   Form submission, plan display, and download
  verify_care_plan.py        Experimental deterministic checks (Layer 1)
  router.py                  Complexity router: patient record → prompt version + model
  generation_loop.py         Evaluator-generator retry loop with hallucination repair
  orchestrator.py            Pure-Python orchestrator combining router and retry loop
  parallel_verifier.py       RacingVerifier, VotingVerifier, SectioningVerifier
  graph_state.py             CarePlanState TypedDict, WorkflowStatus, ErrorRecord
  graph_nodes.py             LangGraph nodes: route, generate, format_check, verify, auto_fix
  graph_edges.py             Conditional edge routing functions
  care_plan_graph.py         StateGraph assembly, compilation, and diagram generation
  templates/care/            Django templates
  test_care_plan_schema.py   Schema validation tests
  test_care_plan_retry.py    Retry-loop tests
  test_structured_llm.py     Provider adapter tests
  test_audit.py              Audit recorder and API tests
  test_router.py             Complexity routing tests
  test_generation_loop.py    Evaluator-generator loop tests
  test_orchestrator.py       Pure-Python orchestrator tests
  test_parallel_verifier.py  Parallel verification strategy tests
  test_graph_scenarios.py    LangGraph end-to-end scenario tests
  query_agents.py            OpenAI Agents SDK query interface (Triage, Progress, Review, Resubmit)
  test_query_routing.py      Intent-routing tests for the query interface
eval/
  dataset.json               Golden cases and stored model runs
  run_eval.py                Evaluation runner
  score_eval.py              Keyword-based scorer
  prompt_benchmark.yaml      Frozen five-case prompt benchmark
  prompt_benchmark.py        Version runner, scorer, and comparison gate
  test_prompt_benchmark.py   Benchmark tests
pharm_project/
  settings.py                Django configuration
  urls.py                    Root routes
prompts/
  manager.py                 Version loading, rendering, and metadata
  config.yaml                Defaults, aliases, versions, and scenarios
  careplan_generation/       Versioned free-text prompts
  careplan_review/           Versioned LLM-as-Judge prompts
  careplan_structured/       Versioned RAG structured-generation prompt
  careplan_repair/           Versioned validation-repair prompt
rag/
  prepare_spl.py             SPL XML cleaning and section-aware chunking
  tests.py                   Chunking tests
DESIGN.md                    Target product design
care_plan_graph.md           Mermaid source for the LangGraph pipeline diagram
care_plan_graph.png          Rendered LangGraph pipeline diagram
docker-compose.yml           Local container setup
Dockerfile                   Application image
```

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Current web MVP | Authenticates Claude requests |
| `ANTHROPIC_MODEL` | No | Claude model override; defaults to `claude-sonnet-4-6` in the structured adapter |
| `OPENAI_API_KEY` | OpenAI adapter and query interface | Authenticates OpenAI requests |
| `OPENAI_MODEL` | No | OpenAI model override; defaults to `gpt-5.5` in the structured adapter |
| `DJANGO_SECRET_KEY` | Production | Django cryptographic signing key |
| `DB_PATH` | No | SQLite path; defaults to `data/db.sqlite3` locally |

## Verification

Run the automated module tests:

```bash
python -m unittest \
  prompts.tests \
  rag.tests \
  care.test_audit \
  care.test_care_plan_schema \
  care.test_structured_llm \
  care.test_care_plan_retry \
  eval.test_prompt_benchmark
```

Run the pure-Python pipeline tests (no Django required):

```bash
python -m pytest \
  care/test_router.py \
  care/test_generation_loop.py \
  care/test_orchestrator.py \
  care/test_parallel_verifier.py \
  -v
```

Run the LangGraph scenario tests (requires the minimal Django configuration
embedded in the test file; no API keys needed):

```bash
python -m pytest care/test_graph_scenarios.py -v
# or run directly for formatted step-by-step output:
python care/test_graph_scenarios.py
```

Run the query interface routing tests (requires `OPENAI_API_KEY`; skipped automatically if not set):

```bash
OPENAI_API_KEY=... python3.11 -m pytest care/test_query_routing.py -v -s
```

Each test prints the responding agent name and final reply so routing decisions
are visible without inspecting internal state.

Run Django's configuration check:

```bash
python manage.py check
```

These tests cover the newer pipeline modules. They do not yet constitute
end-to-end browser, database, retrieval-quality, or clinical-safety testing.

## Safety and data handling

This repository is a prototype and currently lacks several controls expected
for protected health information, including authentication, authorization,
audit-access trails, encryption configuration, retention policies, and a
documented compliant hosting environment. Generation audit records exist, but
they are not a substitute for a complete security and compliance audit
program. Do not use real patient data unless the deployment and operational
environment have been reviewed and approved for that use.

LLM output can be incomplete, incorrect, outdated, or fabricated. A licensed clinician must verify all diagnoses, doses, interactions, contraindications, monitoring recommendations, and patient-specific decisions before use.
