from django.db import models
from django.utils import timezone

JOB_STEP_ORDER = [
    "download",
    "faces",
    "transcription",
    "render",
    "finalize",
]

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
    JOB_TYPE = [
        ("video", "Video"),
        ("story", "Story"),
    ]
    PROCESSING_MODE = [
        ("clips", "Clips"),
        ("full", "Full"),
    ]

    url = models.URLField()
    language = models.CharField(max_length=10, default="auto")  # "pt", "en", "auto"
    status = models.CharField(max_length=20, choices=STATUS, default="pending")
    processing_mode = models.CharField(
        max_length=10,
        choices=PROCESSING_MODE,
        default="clips",
    )
    job_type = models.CharField(
        max_length=10,
        choices=JOB_TYPE,
        default="video",
    )

    title = models.CharField(max_length=255, blank=True)
    source = models.CharField(max_length=20, blank=True)  # yt/tt/ig/other

    original_path = models.CharField(max_length=500, blank=True)  # caminho local do mp4
    error = models.TextField(blank=True)
    story_text = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    transcript_data = models.JSONField(null=True, blank=True)
    transcript_path = models.JSONField(null=True, blank=True)
    video_profile = models.CharField(max_length=50, default="podcast")


class ViralCandidate(models.Model):
    EMOTION = [
        ("curiosity", "Curiosity"),
        ("shock", "Shock"),
        ("opinion", "Opinion"),
        ("neutral", "Neutral"),
    ]

    video_job = models.ForeignKey(
        VideoJob,
        on_delete=models.CASCADE,
        related_name="viral_candidates",
    )
    start_time = models.FloatField()
    end_time = models.FloatField()
    duration = models.FloatField()
    transcript_text = models.TextField()
    viral_score = models.IntegerField()
    emotion = models.CharField(max_length=16, choices=EMOTION, default="neutral")
    reason = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)


class KeywordSignal(models.Model):
    CATEGORY = [
        ("entity", "Entity"),
        ("emotion", "Emotion"),
        ("opinion", "Opinion"),
        ("curiosity", "Curiosity"),
        ("shock", "Shock"),
    ]

    category = models.CharField(max_length=32, choices=CATEGORY)
    term = models.CharField(max_length=120)
    language = models.CharField(max_length=10, default="pt")
    weight = models.IntegerField(default=1)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.term)

    class Meta:
        unique_together = ("category", "term", "language")


class VideoJobStep(models.Model):
    STATUS = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("done", "Done"),
        ("failed", "Failed"),
    ]

    job = models.ForeignKey(VideoJob, on_delete=models.CASCADE, related_name="steps")
    step_name = models.CharField(max_length=64)
    status = models.CharField(max_length=16, choices=STATUS, default="pending")
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)
    message = models.TextField(blank=True)

    class Meta:
        unique_together = ("job", "step_name")


def ensure_job_steps(job: VideoJob, step_names: list[str] | None = None) -> None:
    try:
        steps = step_names or JOB_STEP_ORDER
        for step in steps:
            VideoJobStep.objects.get_or_create(job=job, step_name=step)
    except Exception:
        return None


def update_job_step(
    job_id: int,
    step_name: str,
    status: str,
    message: str | None = None,
) -> None:
    try:
        step, _created = VideoJobStep.objects.get_or_create(
            job_id=job_id,
            step_name=step_name,
        )
        now = timezone.now()
        if status == "running" and step.started_at is None:
            step.started_at = now
        if status in {"done", "failed"}:
            if step.started_at is None:
                step.started_at = now
            if step.finished_at is None:
                step.finished_at = now
            if step.started_at and step.finished_at:
                step.duration_seconds = (
                    step.finished_at - step.started_at
                ).total_seconds()
        step.status = status
        if message is not None:
            step.message = message
        step.save(
            update_fields=[
                "status",
                "started_at",
                "finished_at",
                "duration_seconds",
                "message",
            ]
        )
    except Exception:
        return None


def fail_running_steps(job_id: int, message: str | None = None) -> None:
    try:
        now = timezone.now()
        running_steps = VideoJobStep.objects.filter(job_id=job_id, status="running")
        for step in running_steps:
            if step.started_at is None:
                step.started_at = now
            step.finished_at = now
            step.duration_seconds = (
                step.finished_at - step.started_at
            ).total_seconds()
            step.status = "failed"
            if message is not None:
                step.message = message
            step.save(
                update_fields=[
                    "status",
                    "started_at",
                    "finished_at",
                    "duration_seconds",
                    "message",
                ]
            )
    except Exception:
        return None


def get_job_progress(job: VideoJob) -> dict:
    if job.job_type == "story":
        clips = list(StoryClip.objects.filter(job=job).order_by("part_number"))
        total = len(clips) or 1
        done = len([c for c in clips if c.status == "done"])
        progress_percent = int((done / total) * 100)
        elapsed_seconds = (timezone.now() - job.created_at).total_seconds()
        steps_payload = [
            {
                "name": f"pt{c.part_number}",
                "status": c.status,
                "duration": int(c.duration_seconds) if c.duration_seconds else None,
            }
            for c in clips
        ]
        return {
            "job_id": job.id,
            "status": job.status,
            "current_step": "story",
            "next_step": None,
            "progress_percent": progress_percent,
            "elapsed_seconds": int(elapsed_seconds),
            "estimated_remaining_seconds": None,
            "steps": steps_payload,
        }

    ensure_job_steps(job)
    steps = list(
        VideoJobStep.objects.filter(job=job).order_by("id")
    )
    step_by_name = {s.step_name: s for s in steps}
    ordered_steps = [step_by_name.get(name) for name in JOB_STEP_ORDER]
    ordered_steps = [s for s in ordered_steps if s is not None]

    total_steps = len(ordered_steps) or 1
    done_steps = [s for s in ordered_steps if s.status == "done"]
    running_steps = [s for s in ordered_steps if s.status == "running"]
    current_step = running_steps[0].step_name if running_steps else None

    next_step = None
    for s in ordered_steps:
        if s.status in {"pending", "running"}:
            if current_step is None:
                current_step = s.step_name
            elif s.step_name != current_step and next_step is None:
                next_step = s.step_name
                break
        elif current_step and next_step is None and s.step_name == current_step:
            continue

    progress_percent = int((len(done_steps) / total_steps) * 100)
    elapsed_seconds = (timezone.now() - job.created_at).total_seconds()

    avg_done = None
    if done_steps:
        avg_done = sum(s.duration_seconds or 0 for s in done_steps) / len(done_steps)
    remaining_steps = total_steps - len(done_steps)
    estimated_remaining = None
    if avg_done is not None:
        estimated_remaining = avg_done * remaining_steps

    steps_payload = []
    now = timezone.now()
    for s in ordered_steps:
        duration = s.duration_seconds
        if s.status == "running" and s.started_at:
            duration = (now - s.started_at).total_seconds()
        steps_payload.append(
            {
                "name": s.step_name,
                "status": s.status,
                "duration": int(duration) if duration is not None else None,
            }
        )

    return {
        "job_id": job.id,
        "status": job.status,
        "current_step": current_step,
        "next_step": next_step,
        "progress_percent": progress_percent,
        "elapsed_seconds": int(elapsed_seconds),
        "estimated_remaining_seconds": int(estimated_remaining)
        if estimated_remaining is not None
        else None,
        "steps": steps_payload,
    }

class VideoClip(models.Model):
    CAPTION_STYLE = [
        ("static", "Static"),
        ("word_by_word", "Word by word"),
    ]

    job = models.ForeignKey(VideoJob, on_delete=models.CASCADE, related_name="clips")
    viral_candidate = models.ForeignKey(
        ViralCandidate,
        on_delete=models.SET_NULL,
        related_name="clips",
        null=True,
        blank=True,
    )
    editorial_selected = models.BooleanField(default=False)

    start = models.FloatField()
    end = models.FloatField()
    score = models.FloatField(default=0)

    original_video_path = models.CharField(max_length=500, blank=True)
    original_start = models.FloatField(null=True, blank=True)
    original_end = models.FloatField(null=True, blank=True)
    edited_start = models.FloatField(null=True, blank=True)
    edited_end = models.FloatField(null=True, blank=True)

    caption_style = models.CharField(
        max_length=20,
        choices=CAPTION_STYLE,
        default="static",
    )
    caption_config = models.JSONField(null=True, blank=True)

    caption = models.TextField(blank=True)
    description = models.TextField(blank=True)
    viral_caption = models.TextField(blank=True)
    output_path = models.CharField(max_length=500)  # caminho local do clip final
    thumbnail_path = models.CharField(max_length=500, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def effective_start(self) -> float:
        if self.edited_start is not None:
            return self.edited_start
        if self.original_start is not None:
            return self.original_start
        return self.start

    def effective_end(self) -> float:
        if self.edited_end is not None:
            return self.edited_end
        if self.original_end is not None:
            return self.original_end
        return self.end

    def source_video_path(self) -> str:
        return self.original_video_path or self.job.original_path or ""


class StoryClip(models.Model):
    STATUS = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("done", "Done"),
        ("error", "Error"),
    ]

    job = models.ForeignKey(VideoJob, on_delete=models.CASCADE, related_name="story_clips")
    part_number = models.IntegerField()
    text = models.TextField()
    video_path = models.CharField(max_length=500, blank=True)
    audio_path = models.CharField(max_length=500, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS, default="pending")
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class ClipPublication(models.Model):
    STATUS = [
        ("queued", "Queued"),
        ("publishing", "Publishing"),
        ("published", "Published"),
        ("error", "Error"),
    ]

    PLATFORM = [
        ("youtube", "YouTube"),
    ]

    clip = models.ForeignKey(
        VideoClip,
        on_delete=models.CASCADE,
        related_name="publications",
    )
    platform = models.CharField(max_length=20, choices=PLATFORM)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS, default="queued")
    external_url = models.URLField(blank=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class StoryClipPublication(models.Model):
    STATUS = [
        ("queued", "Queued"),
        ("publishing", "Publishing"),
        ("published", "Published"),
        ("error", "Error"),
    ]

    PLATFORM = [
        ("youtube", "YouTube"),
    ]

    clip = models.ForeignKey(
        "StoryClip",
        on_delete=models.CASCADE,
        related_name="publications",
    )
    platform = models.CharField(max_length=20, choices=PLATFORM)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS, default="queued")
    external_url = models.URLField(blank=True)
    error = models.TextField(blank=True)
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
