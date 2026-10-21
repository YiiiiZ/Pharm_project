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
store. Existing vectors are reused; startup does not ingest or re-embed data.

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

Retrieval filters by the entered medication's product-name substring. No matches
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

## Local approval demo

`DJANGO_DEMO=true` uses a separate `data/demo.sqlite3` database, regardless of
`DB_PATH`, and separate session cookies. Intake and regeneration are disabled.
All seeded plans/evidence are fictional; no model or retrieval calls are made.

```sh
DJANGO_DEMO=true python manage.py migrate
DJANGO_DEMO=true python manage.py seed_approval_demo --password YOUR_DEMO_PASSWORD
DJANGO_DEMO=true python manage.py runserver 127.0.0.1:8000
```

Open http://127.0.0.1:8000/ on this computer. Sign in as `pharmacist` with the
chosen password. The seed command adds three fresh drafts each time, retaining
previous demo decisions. This is a local portal, not a publicly hosted site.

The current demo was started with Gunicorn in daemon mode. Its PID is in
`data/demo-server.pid`; access/error logs are in `data/demo-access.log` and
`data/demo-error.log`. Stop that process before restarting on the same port.

Review verification: `python manage.py test care.test_reviews care.test_jobs care.test_audit`.
