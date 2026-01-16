from django.db import models

class VideoJob(models.Model):
    STATUS = [
        ("pending", "Pending"),
        ("downloading", "Downloading"),
        ("transcribing", "Transcribing"),
        ("clipping", "Clipping"),
        ("done", "Done"),
        ("error", "Error"),
        ("tracking_faces", "Tracking_Faces")
    ]

    url = models.URLField()
    language = models.CharField(max_length=10, default="auto")  # "pt", "en", "auto"
    status = models.CharField(max_length=20, choices=STATUS, default="pending")

    title = models.CharField(max_length=255, blank=True)
    source = models.CharField(max_length=20, blank=True)  # yt/tt/ig/other

    original_path = models.CharField(max_length=500, blank=True)  # caminho local do mp4
    error = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    transcript_data = models.JSONField(null=True, blank=True)
    transcript_path = models.JSONField(null=True, blank=True)

class VideoClip(models.Model):
    job = models.ForeignKey(VideoJob, on_delete=models.CASCADE, related_name="clips")

    start = models.FloatField()
    end = models.FloatField()
    score = models.FloatField(default=0)

    caption = models.TextField(blank=True)
    output_path = models.CharField(max_length=500)  # caminho local do clip final

    created_at = models.DateTimeField(auto_now_add=True)

class ClipFeedback(models.Model):
    clip_id = models.CharField(max_length=64)
    creator_id = models.IntegerField()
    channel = models.CharField(max_length=32)
    retention_3s = models.FloatField()
    retention_50p = models.FloatField()
    retention_100p = models.FloatField()
    hooks = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)