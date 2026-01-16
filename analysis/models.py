# analysis/models.py
from django.db import models

class VideoAnalysis(models.Model):
    platform = models.CharField(max_length=20)  # yt, tt, ig
    query = models.CharField(max_length=255)

    video_id = models.CharField(max_length=100)
    title = models.CharField(max_length=500)
    url = models.URLField()

    views = models.BigIntegerField()
    likes = models.BigIntegerField(null=True, blank=True)
    comments = models.BigIntegerField(null=True, blank=True)

    duration_seconds = models.IntegerField()
    published_at = models.DateTimeField()

    engagement_rate = models.FloatField()
    viral_score = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)
