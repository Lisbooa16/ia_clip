from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("publicar/", views.publishing_guide, name="publishing_guide"),
    path("job/<int:job_id>/", views.job_detail, name="job_detail"),
    path("jobs/<int:job_id>/progress/", views.job_progress, name="job_progress"),
    path("clip/<int:clip_id>/reprocess/", views.reprocess_clip, name="reprocess_clip"),
    path(
        "clip/<int:clip_id>/publish/youtube/",
        views.publish_clip_youtube,
        name="publish_clip_youtube",
    ),
]
