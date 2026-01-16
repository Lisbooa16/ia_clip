from pathlib import Path

from django.conf import settings
from django.http import HttpResponseBadRequest
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST

from .models import VideoJob, VideoClip
from .services import make_vertical_clip_with_captions
from .tasks import process_video_job

def home(request):
    if request.method == "POST":
        url = request.POST.get("url", "").strip()
        language = request.POST.get("language", "auto")

        job = VideoJob.objects.create(url=url, language=language, status="pending")
        process_video_job.apply_async(
            args=[job.id],
            queue="clips_cpu"
        )
        return redirect("job_detail", job_id=job.id)

    jobs = VideoJob.objects.order_by("-created_at")[:20]
    return render(request, "clips/home.html", {"jobs": jobs})

def job_detail(request, job_id):
    job = get_object_or_404(VideoJob, id=job_id)
    return render(request, "clips/job_detail.html", {"job": job})

@require_POST
def reprocess_clip(request, clip_id):
    clip = get_object_or_404(VideoClip, id=clip_id)
    job = clip.job

    media_root = Path(settings.MEDIA_ROOT)

    # 🔑 vídeo original SEMPRE vem do job
    # video_path = media_root / "videos" / "original" / f"{job.original_path}.mp4"

    # transcript você já tem salvo ou pode carregar do JSON
    transcript = job.transcript_data  # veja observação abaixo

    out_mp4, caption = make_vertical_clip_with_captions(
        video_path=str(job.original_path),
        start=clip.start,
        end=clip.end,
        transcript=transcript,
        media_root=media_root,
        clip_id=str(clip.id),
    )

    clip.output_path = out_mp4
    clip.caption = caption
    clip.save(update_fields=["output_path", "caption"])

    return redirect("job_detail", job_id=job.id)