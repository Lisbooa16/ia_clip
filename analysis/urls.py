# analysis/urls.py
from django.urls import path
from .views import analysis_page

urlpatterns = [
    path("analise/", analysis_page, name="analysis_page"),
]
