from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST

from .models import ClipPublication, VideoJob, VideoClip, get_job_progress
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

def publishing_guide(request):
    publishing = settings.SOCIAL_PUBLISHING
    status = {
        platform: {
            "configured": bool(cfg.get("client_id") and cfg.get("client_secret")),
            "note": cfg.get("note", ""),
        }
        for platform, cfg in publishing.items()
    }
    return render(request, "clips/publishing.html", {"publishing_status": status})

def job_detail(request, job_id):
    job = get_object_or_404(VideoJob, id=job_id)
    youtube_cfg = settings.SOCIAL_PUBLISHING.get("youtube", {})
    youtube_ready = bool(
        youtube_cfg.get("client_id") and youtube_cfg.get("client_secret")
    )
    for clip in job.clips.all():
        clip.latest_publication = clip.publications.order_by("-created_at").first()
    return render(
        request,
        "clips/job_detail.html",
        {
            "job": job,
            "youtube_ready": youtube_ready,
        },
    )

def job_progress(request, job_id):
    job = get_object_or_404(VideoJob, id=job_id)
    data = get_job_progress(job)
    if request.GET.get("format") == "html":
        html = f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8" />
            <title>Job {job_id} Progress</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                .bar {{ width: 100%; background: #eee; height: 18px; border-radius: 4px; overflow: hidden; }}
                .bar > div {{ height: 100%; background: #4caf50; width: 0%; }}
                .step {{ margin: 6px 0; }}
                .running {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Job {job_id} Progress</h1>
            <div class="bar"><div id="progress-bar"></div></div>
            <p id="summary"></p>
            <div id="steps"></div>
            <script>
                async function refresh() {{
                    const resp = await fetch("{request.path}?format=json");
                    const data = await resp.json();
                    document.getElementById("progress-bar").style.width = data.progress_percent + "%";
                    document.getElementById("summary").textContent =
                        "Current: " + (data.current_step || "-") +
                        " | Next: " + (data.next_step || "-") +
                        " | Elapsed: " + data.elapsed_seconds + "s";
                    const stepsEl = document.getElementById("steps");
                    stepsEl.innerHTML = "";
                    data.steps.forEach(step => {{
                        const el = document.createElement("div");
                        el.className = "step" + (step.status === "running" ? " running" : "");
                        el.textContent = step.name + " - " + step.status + " (" + (step.duration ?? "-") + "s)";
                        stepsEl.appendChild(el);
                    }});
                }}
                refresh();
                setInterval(refresh, 2000);
            </script>
        </body>
        </html>
        """
        return HttpResponse(html)
    return JsonResponse(data)

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

@require_POST
def publish_clip_youtube(request, clip_id):
    clip = get_object_or_404(VideoClip, id=clip_id)
    job = clip.job

    title = request.POST.get("title", "").strip()
    description = request.POST.get("description", "").strip()

    if not title:
        messages.error(request, "Informe um título para publicar no YouTube.")
        return redirect("job_detail", job_id=job.id)

    youtube_cfg = settings.SOCIAL_PUBLISHING.get("youtube", {})
    if not (youtube_cfg.get("client_id") and youtube_cfg.get("client_secret")):
        messages.error(
            request,
            "Integração do YouTube não configurada. Defina YOUTUBE_CLIENT_ID e "
            "YOUTUBE_CLIENT_SECRET no ambiente.",
        )
        return redirect("job_detail", job_id=job.id)

    ClipPublication.objects.create(
        clip=clip,
        platform="youtube",
        title=title,
        description=description,
        status="queued",
    )
    messages.success(
        request,
        "Solicitação criada. Próximo passo: implementar o upload via YouTube Data API.",
    )
    return redirect("job_detail", job_id=job.id)
