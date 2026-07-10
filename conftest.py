from __future__ import annotations

import os

import django
from django.apps import apps

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pharm_project.settings")

if not apps.ready:
    django.setup()
