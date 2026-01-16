# clips/tasks.py
import json
import uuid
from pathlib import Path
from celery import shared_task, chord
from django.conf import settings

from .focus_strategy import build_focus_timeline
from .media.crop_logic import build_crop_timeline
from .media.face_detection import detect_faces
from .media.face_smoothing import smooth_faces
from .media.face_tracking import track_faces
from .models import (
    VideoJob,
    ensure_job_steps,
    update_job_step,
    fail_running_steps,
)
from .services import (
    download_video,
    transcribe_with_words_to_file,
)
from .services.clip_service import generate_clips
from .tasks_clips import render_clip, finalize_job
from .translator import translate_blueprint_to_cut_plan, generate_clip_sequence


@shared_task(bind=True)
def prepare_face_focus(self, job_id: int):
    job = VideoJob.objects.get(id=job_id)
    media_root = Path(settings.MEDIA_ROOT)
    clip_dir = media_root / "clips" / str(job.id)
    clip_dir.mkdir(parents=True, exist_ok=True)

    try:
        ensure_job_steps(job)
        update_job_step(job.id, "faces", "running")
        # FACE DETECTION (CPU)
        job.status = "detecting_faces"
        job.save(update_fields=["status"])

        fps_sample = 2
        faces = detect_faces(job.original_path, fps_sample=fps_sample)

        # ✅ anti-tremida
        faces_smooth = smooth_faces(faces, alpha=0.75)

        # se não achou nada, não quebra o job inteiro (depende do seu produto)
        if not faces_smooth:
            # salva pra debug e continua (ou raise, você decide)
            empty_path = clip_dir / "faces_smooth.json"
            empty_path.write_text("[]", encoding="utf-8")

        # salvar faces (recomendo salvar só smooth, mas deixo as 2 opções)
        faces_raw_path = clip_dir / "faces_raw.json"
        with open(faces_raw_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fps_sampled": fps_sample,   # ✅ CORRIGIDO
                    "count": len(faces),
                    "faces": faces[:2000],       # ✅ evita arquivo insano (cap de debug)
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        faces_smooth_path = clip_dir / "faces_smooth.json"
        with open(faces_smooth_path, "w", encoding="utf-8") as f:
            json.dump(
                faces_smooth,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # 👇 FACE TRACKING (CPU) — usa o smooth (mais estável)
        job.status = "tracking_faces"
        job.save(update_fields=["status"])

        faces_tracked = track_faces(faces_smooth)

        faces_tracked_path = clip_dir / "faces_tracked.json"
        with open(faces_tracked_path, "w", encoding="utf-8") as f:
            json.dump(
                faces_tracked,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # 👇 CROP TIMELINE (CPU) — usa tracked (ainda melhor) ou smooth
        # Se você não souber frame_w/frame_h, dá pra ler via cv2 aqui.
        frame_w, frame_h = 1920, 1080  # <-- ajuste se seu vídeo não for 1080p

        crop_timeline = build_crop_timeline(
            faces_tracked if faces_tracked else faces_smooth,
            frame_w=frame_w,
            frame_h=frame_h,
            default_hold=1.0,
        )

        crop_timeline_path = clip_dir / "crop_timeline.json"
        with open(crop_timeline_path, "w", encoding="utf-8") as f:
            json.dump(crop_timeline, f, ensure_ascii=False, indent=2)
        update_job_step(job.id, "faces", "done")

    except Exception as e:
        update_job_step(job.id, "faces", "failed", message=str(e))
        job.status = "error"
        job.error = str(e)
        job.save(update_fields=["status", "error"])
        raise


@shared_task(bind=True)
def process_video_job(self, job_id: int):
    job = VideoJob.objects.get(id=job_id)
    media_root = Path(settings.MEDIA_ROOT)

    try:
        ensure_job_steps(job)
        update_job_step(job.id, "download", "running")
        # DOWNLOAD (CPU)
        job.status = "downloading"
        job.save(update_fields=["status"])

        video_path, title = download_video(job.url, media_root, job.source)
        job.original_path = video_path
        job.title = title
        job.save(update_fields=["original_path", "title"])
        update_job_step(job.id, "download", "done")

        clip_dir = media_root / "clips" / str(job.id)
        clip_dir.mkdir(parents=True, exist_ok=True)

        job.status = "queued_processing"
        job.save(update_fields=["status"])

        chord([
            transcribe_video_gpu.s(job.id).set(queue="clips_gpu"),
            prepare_face_focus.s(job.id).set(queue="clips_cpu"),
        ])(
            kickoff_pick_and_render.s(job.id).set(queue="clips_cpu")
        )

    except Exception as e:
        fail_running_steps(job.id, message=str(e))
        job.status = "error"
        job.error = str(e)
        job.save(update_fields=["status", "error"])
        raise

# @shared_task(
#     bind=True,
# )
# def transcribe_video_gpu(self, job_id: int):
#     job = VideoJob.objects.get(id=job_id)
#
#     job.status = "transcribing"
#     job.save(update_fields=["status"])
#
    # transcript = transcribe_with_words(
    #     job.original_path,
    #     language=job.language
    # )
#     print(f"[JOB {job.id}] Transcript gerado ({len(transcript['segments'])} segmentos)")
#
#     job.transcript_data = transcript
#     job.save(update_fields=["transcript_data"])
#
#     print(f"[JOB {job.id}] Transcript salvo")
#
#     pick_and_render.apply_async(
#         args=[job.id],
#         queue="clips_cpu"
#     )
#
#     print(f"[JOB {job.id}] pick_and_render disparado")

@shared_task(bind=True)
def transcribe_video_gpu(self, job_id):
    job = VideoJob.objects.get(id=job_id)

    job.status = "transcribing"
    job.save(update_fields=["status"])

    ensure_job_steps(job)
    update_job_step(job.id, "transcription", "running")

    output_path = Path(settings.MEDIA_ROOT) / "transcripts" / f"{job.id}.json"
    output_path.parent.mkdir(exist_ok=True)

    try:
        result = transcribe_with_words_to_file(
            job.original_path,
            str(output_path),
            language=job.language,
            modelo=job
        )
        # 🔒 GARANTE que o arquivo existe
        if not output_path.exists():
            raise RuntimeError("Transcript file was not created")

        job.transcript_path = str(output_path)
        job.save(update_fields=["transcript_path"])

        print(f"[JOB {job.id}] Transcript salvo em arquivo")
        update_job_step(job.id, "transcription", "done")

        # ✅ RETORNO EXPLÍCITO (CRÍTICO)
        return {
            "job_id": job.id,
            "segments": "written_to_file"
        }
    except Exception as e:
        update_job_step(job.id, "transcription", "failed", message=str(e))
        raise


@shared_task(bind=True)
def kickoff_pick_and_render(self, _results, job_id: int):
    pick_and_render.apply_async(
        args=[job_id],
        queue="clips_cpu"
    )
    return {
        "job_id": job_id,
        "stage": "pick_and_render_queued"
    }


@shared_task(bind=True)
def pick_and_render(self, job_id: int):
    print(f"[PICK] ▶️ Iniciando pick_and_render | job_id={job_id}")

    job = VideoJob.objects.get(id=job_id)

    try:
        # status
        VideoJob.objects.filter(id=job.id).update(status="clipping")
        print(f"[PICK] 📌 Status atualizado para 'clipping'")
        ensure_job_steps(job)
        update_job_step(job.id, "render", "running")

        # 🔥 FONTE ÚNICA DA VERDADE
        if not job.transcript_path:
            print(f"[PICK] ❌ transcript_path vazio no job")
            VideoJob.objects.filter(id=job.id).update(
                status="error",
                error="missing_transcript_path"
            )
            return

        transcript_path = Path(job.transcript_path)
        print(f"[PICK] 📄 transcript_path={transcript_path}")

        if not transcript_path.exists():
            print(f"[PICK] ❌ Arquivo de transcript NÃO encontrado")
            VideoJob.objects.filter(id=job.id).update(
                status="error",
                error="transcript_file_not_found"
            )
            return

        print(f"[PICK] ✅ Arquivo de transcript encontrado")

        # carrega JSON
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        seg_count = len(transcript.get("segments", []))
        print(f"[PICK] 🧠 Transcript carregado | segmentos={seg_count}")

        if seg_count == 0:
            print(f"[PICK] ⚠️ Transcript sem segmentos → finalizando")
            VideoJob.objects.filter(id=job.id).update(status="done")
            return

        # 🔥 FACE FOCUS (ETAPA 3 — DECISÃO GLOBAL)
        clip_dir = Path(settings.MEDIA_ROOT) / "clips" / str(job.id)
        faces_tracked_path = clip_dir / "faces_tracked.json"

        faces_tracked = []
        if faces_tracked_path.exists():
            with open(faces_tracked_path, "r", encoding="utf-8") as f:
                faces_tracked = json.load(f)
        else:
            print("[PICK] ⚠️ faces_tracked.json não encontrado — usando foco central")

        print(f"[PICK] 👤 Faces tracked carregadas: {len(faces_tracked)}")

        focus_timeline = build_focus_timeline(
            faces_tracked=faces_tracked,
            transcript=transcript,
        )

        focus_path = clip_dir / "focus_timeline.json"
        with open(focus_path, "w", encoding="utf-8") as f:
            json.dump(focus_timeline, f, indent=2)

        if focus_timeline:
            focus_path = clip_dir / "focus_timeline.json"
            with open(focus_path, "w", encoding="utf-8") as f:
                json.dump(focus_timeline, f, indent=2)
            print(f"[PICK] 🎯 focus_timeline gerado | blocos={len(focus_timeline)}")
        else:
            print("[PICK] ⚠️ Sem foco por faces: renderizando com crop padrão")

        # picks
        print(f"[PICK] 🔍 Rodando pick_viral_windows...")
        picks = generate_clips(transcript)

        print(f"[PICK] 🎯 Picks encontrados: {len(picks)}")

        if not picks:
            print(f"[PICK] ⚠️ Nenhum pick válido → finalizando")
            VideoJob.objects.filter(id=job.id).update(status="done")
            return

        # cria subtasks
        clip_tasks = []

        for idx, p in enumerate(picks, start=1):
            print(
                f"[PICK] ✂️ Pick {idx}: "
                f"{p['start']:.2f}s → {p['end']:.2f}s | score={p['score']:.2f}"
            )

            clip_tasks.append(
                render_clip.s(
                    job_id=job.id,
                    video_path=job.original_path,
                    transcript=transcript,
                    start=p["start"],
                    end=p["end"],
                    score=p["score"],
                    role=None,
                ).set(queue="clips_cpu")
            )

        print(f"[PICK] 🚀 Disparando {len(clip_tasks)} tasks de render_clip")

        # chord
        chord(clip_tasks)(
            finalize_job.s(job.id).set(queue="clips_cpu")
        )

        print(f"[PICK] 🏁 pick_and_render finalizado com sucesso")
    except Exception as e:
        update_job_step(job.id, "render", "failed", message=str(e))
        raise


@shared_task(bind=True)
def generate_clip_from_blueprint(self, job_id: int, blueprint_path: str):
    job = VideoJob.objects.get(id=job_id)
    media_root = Path(settings.MEDIA_ROOT)

    try:
        ensure_job_steps(job)
        update_job_step(job.id, "download", "running")
        job.status = "downloading"
        job.save(update_fields=["status"])

        video_path, title = download_video(job.url, media_root, job.source)
        job.original_path = video_path
        job.title = title or job.title
        job.save(update_fields=["original_path", "title"])
        update_job_step(job.id, "download", "done")

        update_job_step(job.id, "transcription", "running")
        job.status = "transcribing"
        job.save(update_fields=["status"])

        transcript_path = media_root / "transcripts" / f"{job.id}.json"
        transcript_path.parent.mkdir(exist_ok=True)
        transcribe_with_words_to_file(
            job.original_path,
            str(transcript_path),
            language=job.language,
            modelo=job,
        )

        job.transcript_path = str(transcript_path)
        job.save(update_fields=["transcript_path"])
        update_job_step(job.id, "transcription", "done")

        blueprint = json.loads(Path(blueprint_path).read_text(encoding="utf-8"))
        blueprint_data = blueprint.get("blueprint", {})

        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))

        clip_dir = media_root / "clips" / str(job.id)
        clip_dir.mkdir(parents=True, exist_ok=True)

        update_job_step(job.id, "faces", "running")
        fps_sample = 1
        faces = detect_faces(job.original_path, fps_sample=fps_sample)
        faces_smooth = smooth_faces(faces, alpha=0.75)
        faces_tracked = track_faces(faces_smooth) if faces_smooth else []

        faces_tracked_path = clip_dir / "faces_tracked.json"
        with open(faces_tracked_path, "w", encoding="utf-8") as f:
            json.dump(faces_tracked, f, ensure_ascii=False, indent=2)

        focus_timeline = build_focus_timeline(
            faces_tracked=faces_tracked,
            transcript=transcript,
        )
        focus_path = clip_dir / "focus_timeline.json"
        with open(focus_path, "w", encoding="utf-8") as f:
            json.dump(focus_timeline, f, indent=2)
        update_job_step(job.id, "faces", "done")

        cut_plans = generate_clip_sequence(transcript, blueprint_data, job.source or "other")
        if not cut_plans:
            cut_plans = [translate_blueprint_to_cut_plan(transcript, blueprint_data)]

        job.status = "clipping"
        job.save(update_fields=["status"])
        update_job_step(job.id, "render", "running")

        clip_tasks = []
        for plan in cut_plans:
            clip_tasks.append(
                render_clip.s(
                    job_id=job.id,
                    video_path=job.original_path,
                    transcript=transcript,
                    start=plan["start"],
                    end=plan["end"],
                    score=0,
                    role=plan.get("role"),
                ).set(queue="clips_cpu")
            )

        chord(clip_tasks)(
            finalize_job.s(job.id).set(queue="clips_cpu")
        )

    except Exception as e:
        fail_running_steps(job.id, message=str(e))
        job.status = "error"
        job.error = str(e)
        job.save(update_fields=["status", "error"])
        raise
