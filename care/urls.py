from django.urls import path
from . import views

urlpatterns = [
    path("", views.order_form, name="order_form"),
    path("order/<int:pk>/", views.care_plan, name="care_plan"),
    path("order/<int:pk>/download/", views.download_care_plan, name="download_care_plan"),
]
