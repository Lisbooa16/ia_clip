import re

from django.conf import settings
from django.contrib import messages
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST

from .models import (
    ClipPublication,
    StoryClipPublication,
    VideoJob,
    VideoClip,
    ViralCandidate,
    get_job_progress,
)
from .tasks import process_video_job, publish_clip_to_youtube, process_story_job, publish_story_clip_to_youtube
from .tasks_clips import render_clip_edit

from django.utils.dateparse import parse_datetime
from django.utils.timezone import make_aware, get_current_timezone
from datetime import timezone

from .text_story import split_story_text
from .services.analysis import load_transcript_for_job, split_transcript_into_candidates, score_candidate
from .services.copywriter import generate_youtube_description, generate_viral_caption


def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _render_clip_from_edit(clip: VideoClip) -> None:
    render_clip_edit.apply_async(args=[clip.id], queue="clips_cpu")

def home(request):
    if request.method == "POST":
        # Video ingestion entrypoint: only enqueue process_video_job here.
        job_type = request.POST.get("job_type", "video")
        url = request.POST.get("url", "").strip()
        language = request.POST.get("language", "auto")
        processing_mode = request.POST.get("processing_mode", "clips")
        story_title = request.POST.get("story_title", "").strip()
        story_text = request.POST.get("story_text", "").strip()

        if job_type == "story":
            if not story_text:
                messages.error(request, "Informe o texto da história.")
                return redirect("home")
            job = VideoJob.objects.create(
                url="",
                language=language,
                processing_mode="clips",
                status="pending",
                job_type="story",
                title=story_title,
                story_text=story_text,
            )
            process_story_job.apply_async(
                args=[job.id],
                queue="clips_cpu",
            )
            return redirect("job_detail", job_id=job.id)

        if not url:
            messages.error(request, "Informe a URL do vídeo.")
            return redirect("home")

        job = VideoJob.objects.create(
            url=url,
            language=language,
            processing_mode=processing_mode,
            status="pending",
            job_type="video",
        )
        process_video_job.apply_async(
            args=[job.id],
            queue="clips_cpu"
        )
        return redirect("analysis_view", job_id=job.id)

    jobs = VideoJob.objects.order_by("-created_at")[:20]
    return render(request, "clips/home.html", {"jobs": jobs})


def text_story_create(request):
    if request.method == "POST":
        story_text = request.POST.get("story_text", "").strip()
        story_title = request.POST.get("story_title", "").strip()

        if not story_text:
            messages.error(request, "Informe o texto da história.")
            return redirect("text_story_create")

        parts = split_story_text(story_text)
        if not parts:
            messages.error(request, "Texto inválido para dividir em partes.")
            return redirect("text_story_create")

        job = VideoJob.objects.create(
            url="",
            language="pt",
            processing_mode="clips",
            status="pending",
            job_type="story",
            title=story_title,
            story_text=story_text,
            source="text_story",
        )
        process_story_job.apply_async(
            args=[job.id],
            queue="clips_cpu",
        )
        messages.success(request, "Job de hist¢ria enfileirado para render.")
        return redirect("job_detail", job_id=job.id)

    return render(request, "clips/text_story_create.html")

def publishing_guide(request):
    publishing = settings.SOCIAL_PUBLISHING
    readiness = {
        "youtube": ["client_id", "client_secret", "refresh_token"],
        "instagram": ["app_id", "app_secret"],
        "tiktok": ["client_key", "client_secret"],
    }
    status = {
        platform: {
            "configured": bool(cfg.get("client_id") and cfg.get("client_secret")),
            "configured": bool(
                all(cfg.get(field) for field in readiness.get(platform, []))
            ),
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
    youtube_ready = True
    for clip in job.clips.all():
        clip.latest_publication = clip.publications.order_by("-created_at").first()
        try:
            clip.duration_seconds = max(0.0, clip.effective_end() - clip.effective_start())
        except Exception:
            clip.duration_seconds = None
    story_clips = []
    total_story_duration = None
    if job.job_type == "story":
        story_clips = list(job.story_clips.order_by("part_number"))
        for clip in story_clips:
            clip.latest_publication = clip.publications.order_by("-created_at").first()
        durations = [c.duration_seconds or 0 for c in story_clips]
        total_story_duration = sum(durations) if durations else None
    full_video_clip = None
    if job.processing_mode == "full":
        full_video_clip = job.clips.filter(caption="Vídeo completo").order_by("id").first()
    return render(
        request,
        "clips/job_detail.html",
        {
            "job": job,
            "youtube_ready": youtube_ready,
            "full_video_clip": full_video_clip,
            "story_clips": story_clips,
            "total_story_duration": total_story_duration,
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
    _render_clip_from_edit(clip)
    messages.success(request, "Reprocessamento enfileirado.")
    return redirect("job_detail", job_id=clip.job_id)

@require_POST
def update_clip_edit(request, clip_id):
    clip = get_object_or_404(VideoClip, id=clip_id)

    edited_start = _parse_float(request.POST.get("edited_start"))
    edited_end = _parse_float(request.POST.get("edited_end"))
    caption_style = (request.POST.get("caption_style") or "").strip().lower()
    if caption_style not in {"static", "word_by_word"}:
        caption_style = clip.caption_style or "static"

    caption_config = {
        "font_family": (request.POST.get("font_family") or "").strip() or None,
        "font_size": _parse_float(request.POST.get("font_size")),
        "font_color": (request.POST.get("font_color") or "").strip() or None,
        "highlight_color": (request.POST.get("highlight_color") or "").strip() or None,
        "background": bool(request.POST.get("background")),
        "position": (request.POST.get("position") or "").strip() or None,
    }
    caption_config = {k: v for k, v in caption_config.items() if v is not None}

    clip.edited_start = edited_start
    clip.edited_end = edited_end
    clip.caption_style = caption_style
    clip.caption_config = caption_config or None
    clip.save(update_fields=[
        "edited_start",
        "edited_end",
        "caption_style",
        "caption_config",
    ])

    _render_clip_from_edit(clip)
    messages.success(request, "Edição salva. Reprocessamento enfileirado.")
    return redirect("job_detail", job_id=clip.job_id)

@require_POST
def publish_clip_youtube(request, clip_id):
    clip = get_object_or_404(VideoClip, id=clip_id)
    job = clip.job

    title = request.POST.get("title", "").strip()
    description = request.POST.get("description", "").strip()
    youtube_channel = request.POST.get("youtube_channel")
    publish_at_raw = request.POST.get("publish_at")

    if not title:
        messages.error(request, "Informe um título para publicar no YouTube.")
        return redirect("job_detail", job_id=job.id)

    if job.source == "text_story" and job.title:
        match = re.search(r"\bPT(\d+)\b", job.title, re.IGNORECASE)
        if match:
            pt_label = f"PT{match.group(1)}"
            if pt_label.lower() not in title.lower():
                title = f"{title} {pt_label}".strip()

    channels = settings.SOCIAL_PUBLISHING.get("youtube", {}).get("channels", {})
    if youtube_channel not in channels:
        messages.error(request, "Canal do YouTube inválido.")
        return redirect("job_detail", job_id=job.id)

    publish_at = None
    if publish_at_raw:
        dt = parse_datetime(publish_at_raw)
        if dt:
            publish_at = make_aware(dt, get_current_timezone()).astimezone(timezone.utc)

    publication = ClipPublication.objects.create(
        clip=clip,
        platform="youtube",
        title=title,
        description=description,
        status="queued",
    )

    # 👇 PASSA O CANAL DIRETO PRA TASK
    publish_clip_to_youtube.apply_async(
        args=[publication.id, youtube_channel, publish_at.isoformat() if publish_at else None],
        queue="clips_cpu",
    )

    messages.success(
        request,
        f"Publicação enfileirada para o canal: {youtube_channel}",
    )
    return redirect("job_detail", job_id=job.id)


@require_POST
def publish_story_job(request, job_id):
    job = get_object_or_404(VideoJob, id=job_id)
    if job.job_type != "story":
        return HttpResponseBadRequest("Job inválido.")

    title_base = request.POST.get("title", "").strip() or job.title or "História"
    description = request.POST.get("description", "").strip()
    youtube_channel = request.POST.get("youtube_channel")
    publish_at_raw = request.POST.get("publish_at")

    channels = settings.SOCIAL_PUBLISHING.get("youtube", {}).get("channels", {})
    if youtube_channel not in channels:
        messages.error(request, "Canal do YouTube inválido.")
        return redirect("job_detail", job_id=job.id)

    publish_at = None
    if publish_at_raw:
        dt = parse_datetime(publish_at_raw)
        if dt:
            publish_at = make_aware(dt, get_current_timezone()).astimezone(timezone.utc)

    story_clips = job.story_clips.order_by("part_number")
    for clip in story_clips:
        if not clip.video_path:
            continue
        publication = StoryClipPublication.objects.create(
            clip=clip,
            platform="youtube",
            title=f"{title_base} PT{clip.part_number}",
            description=description,
            status="queued",
        )
        publish_story_clip_to_youtube.apply_async(
            args=[publication.id, youtube_channel, publish_at.isoformat() if publish_at else None],
            queue="clips_cpu",
        )

    messages.success(request, "Publicações enfileiradas para a história.")
    return redirect("job_detail", job_id=job.id)


def analysis_view(request, job_id):
    job = get_object_or_404(VideoJob, id=job_id)
    candidates = list(
        ViralCandidate.objects.filter(video_job=job).order_by("start_time")
    )
    transcript_ready = bool(load_transcript_for_job(job))
    return render(
        request,
        "clips/analysis.html",
        {
            "job": job,
            "candidates": candidates,
            "transcript_ready": transcript_ready,
        },
    )


@require_POST
def run_viral_analysis(request, job_id):
    job = get_object_or_404(VideoJob, id=job_id)
    if job.job_type != "video":
        messages.error(request, "Análise viral disponível apenas para vídeos.")
        return redirect("analysis_view", job_id=job.id)
    transcript = load_transcript_for_job(job)
    if not transcript:
        messages.error(request, "Transcript indisponível para análise.")
        return redirect("analysis_view", job_id=job.id)

    ViralCandidate.objects.filter(video_job=job).delete()
    candidates = split_transcript_into_candidates(transcript)
    created = 0
    for candidate in candidates:
        score, emotion, reason = score_candidate(candidate["text"], candidate["duration"])
        ViralCandidate.objects.create(
            video_job=job,
            start_time=round(candidate["start"], 3),
            end_time=round(candidate["end"], 3),
            duration=round(candidate["duration"], 3),
            transcript_text=candidate["text"],
            viral_score=score,
            emotion=emotion,
            reason=reason,
        )
        created += 1

    if created:
        messages.success(request, f"Análise gerou {created} candidatos.")
    else:
        messages.error(request, "Nenhum candidato encontrado com as regras atuais.")
    return redirect("analysis_view", job_id=job.id)
