from django.contrib import messages
from django.shortcuts import redirect, render, get_object_or_404
from django.views.decorators.http import require_POST

from clips.models import VideoClip, ViralCandidate
from clips.services.analysis import load_transcript_for_job
from clips.services.copywriter import generate_youtube_description, generate_viral_caption
from clips.services.clip_generator import build_editorial_overlays
from clips.tasks_clips import render_clip

from .services.viral_analysis import build_analysis


def analysis_page(request):
    result = None
    url = ""
    error = ""

    if request.method == "POST":
        url = request.POST.get("url", "").strip()

        if url:
            result = build_analysis(url).to_dict()
        else:
            error = "Informe uma URL válida."

    return render(
        request,
        "analysis/analysis.html",
        {
            "url": url,
            "result": result,
            "error": error,
        },
    )


@require_POST
def generate_clip(request):
    # Clip generation entrypoint: only creates a single clip from a ViralCandidate.
    candidate_id = request.POST.get("candidate_id")
    if not candidate_id:
        messages.error(request, "Selecione um candidato antes de gerar o clip.")
        return redirect("analysis_page")

    candidate = get_object_or_404(ViralCandidate, id=candidate_id)
    job = candidate.video_job

    existing = VideoClip.objects.filter(viral_candidate=candidate).order_by("-id").first()
    if existing:
        messages.info(request, "Clip já existe para este candidato.")
        return redirect("analysis_view", job_id=job.id)

    if not job.original_path:
        messages.error(request, "Vídeo original indisponível para gerar o clip.")
        return redirect("analysis_view", job_id=job.id)

    transcript = load_transcript_for_job(job)
    if not transcript:
        messages.error(request, "Transcript indisponível para gerar o clip.")
        return redirect("analysis_view", job_id=job.id)

    overlays = build_editorial_overlays(
        candidate.transcript_text,
        candidate.start_time,
        candidate.end_time,
        title=job.title or None,
    )

    clip = VideoClip.objects.create(
        job=job,
        start=candidate.start_time,
        end=candidate.end_time,
        original_start=candidate.start_time,
        original_end=candidate.end_time,
        original_video_path=job.original_path,
        score=candidate.viral_score,
        caption="",
        output_path="",
        viral_candidate=candidate,
        editorial_selected=True,
    )
    description = generate_youtube_description(clip, candidate)
    viral_caption = generate_viral_caption(clip, candidate)
    clip.description = description
    clip.viral_caption = viral_caption
    update_fields = ["description", "viral_caption"]
    if not clip.caption:
        clip.caption = viral_caption
        update_fields.append("caption")
    clip.save(update_fields=update_fields)

    render_clip.apply_async(
        args=[
            job.id,
            job.original_path,
            transcript,
            candidate.start_time,
            candidate.end_time,
            float(candidate.viral_score),
            "editorial",
        ],
        kwargs={
            "clip_id": clip.id,
            "overlay_segments": overlays,
        },
        queue="clips_cpu",
    )

    messages.success(request, "Clip editorial enfileirado.")
    return redirect("analysis_view", job_id=job.id)
