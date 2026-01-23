from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("text-story/create/", views.text_story_create, name="text_story_create"),
    path("publicar/", views.publishing_guide, name="publishing_guide"),
    path("job/<int:job_id>/", views.job_detail, name="job_detail"),
    path("jobs/<int:job_id>/progress/", views.job_progress, name="job_progress"),
    path("clip/<int:clip_id>/reprocess/", views.reprocess_clip, name="reprocess_clip"),
    path("clip/<int:clip_id>/edit/", views.update_clip_edit, name="update_clip_edit"),
    path("job/<int:job_id>/publish/story/youtube/", views.publish_story_job, name="publish_story_job"),
    path(
        "clip/<int:clip_id>/publish/youtube/",
        views.publish_clip_youtube,
        name="publish_clip_youtube",
    ),
]
