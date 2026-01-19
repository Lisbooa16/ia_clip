import json
from pathlib import Path

from django.conf import settings
from django.shortcuts import redirect, render

from clips.models import VideoJob
from clips.services import detect_source
from clips.tasks import generate_clip_from_blueprint, generate_viral_clips

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


def generate_clip(request):
    if request.method != "POST":
        return redirect("analysis_page")

    url = request.POST.get("url", "").strip()
    idea = request.POST.get("idea", "").strip()
    mode = request.POST.get("mode", "manual").strip()  # "manual" ou "viral"
    profile = request.POST.get("profile", "podcast").strip()

    if not url:
        return redirect("analysis_page")

    source = detect_source(url)
    job = VideoJob.objects.create(
        url=url,
        language="auto",
        status="pending",
        title=idea[:255] or "Clip Viral",
        source=source,
    )

    # 🔥 MODO VIRAL (automático com hooks)
    if mode == "viral":
        generate_viral_clips.apply_async(
            args=[job.id],
            kwargs={"profile": profile},
            queue="clips_cpu",
        )
        return redirect("job_detail", job_id=job.id)

    # 📝 MODO MANUAL (blueprint)
    blueprint = {
        "opening": request.POST.get("opening", "").strip(),
        "setup": request.POST.get("setup", "").strip(),
        "context": request.POST.get("context", "").strip(),
        "tension": request.POST.get("tension", "").strip(),
        "reveal": request.POST.get("reveal", "").strip(),
        "ending": request.POST.get("ending", "").strip(),
    }

    if not blueprint["opening"]:
        return redirect("analysis_page")

    clip_dir = Path(settings.MEDIA_ROOT) / "clips" / str(job.id)
    clip_dir.mkdir(parents=True, exist_ok=True)
    blueprint_path = clip_dir / "blueprint.json"
    blueprint_path.write_text(
        json.dumps(
            {
                "idea": idea,
                "blueprint": blueprint,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    generate_clip_from_blueprint.apply_async(
        args=[job.id, str(blueprint_path)],
        queue="clips_cpu",
    )

    return redirect("job_detail", job_id=job.id)