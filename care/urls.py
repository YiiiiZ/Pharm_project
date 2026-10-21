from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path("jobs/<uuid:pk>/review/", views.submit_review, name="submit_review"),
    path("login/", auth_views.LoginView.as_view(template_name="care/login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("api/jobs/<uuid:pk>/", views.job_status, name="job_status"),
    path("order/<int:pk>/retry/", views.retry_generation, name="retry_generation"),
    path("", views.order_form, name="order_form"),
    path("api/llm-logs", views.llm_logs, name="llm_logs"),
    path("order/<int:pk>/", views.care_plan, name="care_plan"),
    path("order/<int:pk>/download/", views.download_care_plan, name="download_care_plan"),
]
