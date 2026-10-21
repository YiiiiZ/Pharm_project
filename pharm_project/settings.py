import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-secret-key-change-in-prod")
DEBUG = os.environ.get("DJANGO_DEBUG", "false").lower() == "true"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]").split(",")

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.sessions",
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "care",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
]

ROOT_URLCONF = "pharm_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "pharm_project.wsgi.application"

DEMO_MODE = os.environ.get("DJANGO_DEMO", "false").lower() == "true"

DB_PATH = os.environ.get("DB_PATH", str(BASE_DIR / "data" / "db.sqlite3"))
if DEMO_MODE:
    DB_PATH = str(BASE_DIR / "data" / "demo.sqlite3")

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": DB_PATH,
    }
}

STATIC_URL = "/static/"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/login/"

SESSION_COOKIE_NAME = "pharm_demo_sessionid" if DEMO_MODE else "sessionid"
CSRF_COOKIE_NAME = "pharm_demo_csrftoken" if DEMO_MODE else "csrftoken"
