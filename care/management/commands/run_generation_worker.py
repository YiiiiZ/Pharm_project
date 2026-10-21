import time
from django.core.management.base import BaseCommand
from django.db import close_old_connections
from care.worker import claim_next, run_job, fail_stale_jobs


class Command(BaseCommand):
    help = "Process durable generation jobs. Run one worker for the SQLite MVP."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true", help="Process at most one job, then exit")
        parser.add_argument("--recover-stale", action="store_true", help="Mark jobs idle for 30 minutes failed; use after stopping the old worker")

    def handle(self, *args, **options):
        if options["recover_stale"]:
            self.stdout.write(f"Recovered {fail_stale_jobs()} interrupted jobs")
            return
        try:
            while True:
                close_old_connections()
                job = claim_next()
                if job:
                    run_job(job)
                if options["once"]:
                    return
                if not job:
                    time.sleep(2)
        except KeyboardInterrupt:
            self.stdout.write("Worker stopped")
