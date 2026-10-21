# Running the connected MVP

## Scope

Django staff login → validated order → database job queue → worker → label
retrieval → structured Claude generation → schema and deterministic checks →
saved draft, evidence, and report → status/review page.

The shared application operations live in `care/services.py`. A `GenerationJob`
contains one input snapshot and one run's result. Failed retries create another
job and retain previous results. Audit `run_group_id` matches the job UUID.
The database queue avoids another broker dependency and commits the order and
job together. Run **one worker** with SQLite; PostgreSQL operational storage,
chat integration and clinical LLM review are deferred. Pharmacist decisions are supported.

The limited checks compare patient summary fields and require supplied source
IDs on problems/goals/interventions. They do not establish clinical correctness
or claim-to-source support. High-risk patient mismatches are shown for pharmacist
review; repeated citation/format failures terminate as failed. A draft is only marked approved after an authorized pharmacist explicitly approves it. Duplicate request tokens prevent repeated form
submission from creating another order. Broader clinical duplicate-order rules
are not implemented.

## Local setup

Use Python 3.11 (matching Docker) or Python 3.12 (validated locally), install `requirements.txt`, and export the
settings in `.env.example` from your own `.env`. Set a strong Django secret.
Both `ANTHROPIC_API_KEY` (generation) and `OPENAI_API_KEY` (query embeddings) are
needed by the worker. `VECTOR_DATABASE_URL` must point to the populated vector
store. Existing vectors are reused. Optionally set `SPL_CHUNKS_PATH` to a prepared
JSONL corpus: the worker indexes matching product/generic labels before retrieval,
reuses existing vectors by content hash, and resumes incomplete indexing on retry.
The first request for a medication may take longer. No full-corpus indexing occurs
at startup; without this setting, populate the vector store using `rag.embed`.

Before migrating an existing installation, stop the old web process and back up
its SQLite database and vector volume. The initial migration represents the
previous Provider/Patient/Order/GenerationAudit tables. `--fake-initial` adopts
those tables only when they already exist; the second migration adds jobs. If
only some old tables exist, reconcile that older schema before proceeding.

```sh
mkdir -p data
python manage.py migrate --fake-initial
python manage.py createsuperuser
python manage.py runserver 127.0.0.1:8000
```

In a second shell with the same environment and database:

```sh
python manage.py run_generation_worker
```

Open `http://127.0.0.1:8000/` and sign in. Other accounts need `is_staff=True`.
All staff have access to the internal order workspace in this MVP.
`DJANGO_ALLOWED_HOSTS` must include any additional hostname you use.

Retrieval filters by the entered medication's product or generic-name substring. No matches
produce a visible failure instead of silently generating without evidence.
Use a product name present in the index; automatic brand/generic alias resolution
is deferred. Clinical notes are not sent to the embedding provider.

## Docker

```sh
docker compose up --build -d
docker compose exec web python manage.py createsuperuser
```

The web service migrates before Gunicorn starts. The worker waits for a healthy
web service and uses the same SQLite volume. It accesses `vector_db` internally.
The existing vector volume is preserved. No production deployment is implied.

## Failure recovery

A stopped worker leaves queued jobs in the database. Restart it to process them.
A job interrupted while running is not automatically replayed: first stop the
old worker, then mark runs idle for at least 30 minutes as failed:

```sh
python manage.py run_generation_worker --recover-stale
```

Restart the worker and use **Retry generation** on the failed order. Conditional
updates prevent an obsolete worker from publishing over a recovered job. An
external API request already in flight may still finish and be billed.

For one queued job only:

```sh
python manage.py run_generation_worker --once
```

The worker logs the job ID and exception type without dumping patient/provider
responses. Detailed LLM failures remain in the staff-only generation audit.

## Tests

```sh
python manage.py test care.test_jobs care.test_audit care.test_structured_llm care.test_care_plan_retry care.test_router care.test_orchestrator care.test_generation_loop care.test_parallel_verifier care.test_care_plan_schema prompts eval.test_prompt_benchmark
python -m pytest care/test_graph_scenarios.py -q
```

These tests mock external providers. Live retrieval and agent-routing tests are
separate and can make paid calls if keys are present. `rag/tests.py` currently
imports `rag.prepare_spl`, which was already deleted in the working tree before
this integration; the ingestion-source change has been left untouched.

The existing local `venv` uses Python 3.14, which is incompatible with Django
5.0 template testing. Use Docker or recreate a development environment with
Python 3.11/3.12 rather than reusing that environment.


## Pharmacist review

A completed draft can be approved, rejected, or returned for changes. Comments
are required for rejection and changes requests. Each version has one final
review decision with reviewer identity, timestamp, comments, and the plan's
SHA-256 checksum. Decisions cannot be overwritten through the application.
A rejected/changes-requested plan can generate a new draft; its old review is
retained and the new draft requires its own review. Approved downloads include
the reviewer and decision date. Approval does not change the generation status;
the job API exposes a separate `review_status`.

Run migrations before using review. Existing staff accounts do not automatically
receive clinical review permission. A superuser already has permission; grant
`care.review_careplan` to other pharmacist accounts using Django's permission
API in `python manage.py shell`:

```python
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
user = get_user_model().objects.get(username="YOUR_PHARMACIST_USERNAME")
user.is_staff = True
user.save(update_fields=["is_staff"])
user.user_permissions.add(Permission.objects.get(
    content_type__app_label="care", codename="review_careplan"))
```

## Live local pipeline demo

`DJANGO_DEMO=true` uses a separate `data/demo.sqlite3` database, regardless of
`DB_PATH`, and separate session cookies. The demo now supports the full live flow:
patient intake → durable queue → medication-label indexing/retrieval → LangGraph
routing and Claude generation → schema validation → patient/citation checks and
bounded repair → **Waiting for pharmacist review**. It never auto-approves a plan.

The current portal is http://127.0.0.1:8000/. Sign in, click **Load fictional
example**, inspect or edit the patient details, then click **Generate care plan**.
The example uses metformin, which is present in the local label corpus. The next
page refreshes as stages change and retains the progress history, evidence,
structured draft and check report. Retrieval covers indication, dosage and safety
queries and deduplicates their evidence by content hash. Review controls appear only on a completed
draft. A medication with no matching label evidence fails visibly without
calling the generation model. Label matching is substring-based, not terminology
or formulation reconciliation; review the retrieved product/formulation carefully.

Real API calls are made and billed: OpenAI embeds label/query text; Anthropic
receives the submitted patient snapshot and retrieved evidence. Use fictional
patient details in this demo. The older seeded approval examples remain available;
their evidence/plans are fictional and were not generated through the live pipeline.

To restart locally, activate a Python 3.11/3.12 environment with requirements
installed. Load your `.env` into the shell, then run:

```sh
set -a
. ./.env
set +a
export DJANGO_DEMO=true
export SPL_CHUNKS_PATH="$PWD/data/rag/spl_chunks.jsonl"
docker compose up -d vector_db
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

Use `VECTOR_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/spl_vectors`
for local processes. In a second shell with the same environment, run
`python manage.py run_generation_worker`. Run only one worker with SQLite.
The prepared corpus must already exist at `SPL_CHUNKS_PATH`; unset this variable
when using a previously populated vector store without a local corpus.
On a fresh demo database create a staff account using `createsuperuser` (existing
local accounts remain available). `seed_approval_demo --password YOUR_DEMO_PASSWORD`
is optional and creates only the older fictional approval examples.

The current web server runs in Gunicorn daemon mode, with PID
`data/demo-server.pid` and logs `data/demo-access.log`, `data/demo-error.log`.
The current background worker PID is `data/demo-worker.pid`, with log
`data/demo-worker.log`. Stop those processes before starting replacements; avoid
interrupting an active job. The vector database runs as the compose `vector_db`
service and retains its named volume. These local files and `.env` are ignored
by Git. This portal is local to this computer, not publicly hosted.

Verification:

```sh
python manage.py test care.test_reviews care.test_jobs care.test_audit
python -m unittest rag.test_embedding_cache
```

### Additional local staff account

The current local demo also has this account:

- Username: `yizheng`
- Login: http://127.0.0.1:8000/login/
- Access: active staff account with `care.review_careplan` permission, allowing
  approval, rejection, and requests for changes.
- Database: the isolated `data/demo.sqlite3` demo database.

Use the password configured locally for this account. Login credentials are not
recorded in this repository. The account was created locally; cloning the
repository or running `seed_approval_demo` does not recreate `yizheng`.
