from django.urls import path
from .views import analysis_page, generate_clip

urlpatterns = [
    path("analise/", analysis_page, name="analysis_page"),
    path("analise/gerar-clip/", generate_clip, name="analysis_generate_clip"),
]
