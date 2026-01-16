from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("job/<int:job_id>/", views.job_detail, name="job_detail"),
    path("clip/<int:clip_id>/reprocess/", views.reprocess_clip, name="reprocess_clip"),
]
