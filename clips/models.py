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

    title = models.CharField(max_length=255, blank=True)
    source = models.CharField(max_length=20, blank=True)  # yt/tt/ig/other

    original_path = models.CharField(max_length=500, blank=True)  # caminho local do mp4
    error = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    transcript_data = models.JSONField(null=True, blank=True)
    transcript_path = models.JSONField(null=True, blank=True)
    video_profile = models.CharField(max_length=50, default="podcast")


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
    job = models.ForeignKey(VideoJob, on_delete=models.CASCADE, related_name="clips")

    start = models.FloatField()
    end = models.FloatField()
    score = models.FloatField(default=0)

    caption = models.TextField(blank=True)
    output_path = models.CharField(max_length=500)  # caminho local do clip final

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

class ClipFeedback(models.Model):
    clip_id = models.CharField(max_length=64)
    creator_id = models.IntegerField()
    channel = models.CharField(max_length=32)
    retention_3s = models.FloatField()
    retention_50p = models.FloatField()
    retention_100p = models.FloatField()
    hooks = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
