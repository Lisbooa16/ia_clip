import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
import cv2
from celery import shared_task, chord
from django.conf import settings

from analysis.services.viral_analysis import _extract_keywords, _classify_story
from .focus_strategy import build_focus_timeline
from .media.crop_logic import build_crop_timeline
from .media.face_detection import detect_faces
from .media.face_smoothing import smooth_faces
from .media.face_tracking import track_faces
from .models import (
    ClipPublication,
    StoryClipPublication,
    VideoJob,
    VideoClip,
    StoryClip,
    ensure_job_steps,
    update_job_step,
    fail_running_steps,
)
from subtitles.subtitle_builder import fill_gaps, to_srt
from .services import (
    build_static_background_video,
    build_words_timeline,
    download_video,
    make_vertical_clip_with_captions,
    transcribe_with_words_to_file,
    FFMPEG_BIN,
    pick_viral_windows_rich,
    pick_viral_windows,
)
from .services.clip_service import generate_clips
from .tasks_clips import render_clip, finalize_job
from .text_story import build_transcript_from_text
from .translator import translate_blueprint_to_cut_plan, generate_clip_sequence
from .services.youtube_service import upload_clip_publication, upload_story_publication
from subtitles.subtitle_builder import build_word_by_word_ass
from subtitles.caption_styles import CaptionStyleConfig, CaptionPosition
import re
import math

def _faces_cache_key(video_path: str, fps_sample: int) -> str | None:
    try:
        stat = os.stat(video_path)
        base = f"{video_path}|{fps_sample}|{stat.st_mtime_ns}|{stat.st_size}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()
    except OSError:
        return None


def _load_faces_cache(media_root: Path, video_path: str, fps_sample: int):
    try:
        key = _faces_cache_key(video_path, fps_sample)
        if not key:
            return None
        cache_path = media_root / "faces_cache" / f"{key}.json"
        if not cache_path.exists():
            return None
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("fps_sample") != fps_sample:
            return None
        return payload
    except Exception:
        return None


def _save_faces_cache(
    media_root: Path,
    video_path: str,
    fps_sample: int,
    faces: list,
    faces_smooth: list,
) -> None:
    try:
        key = _faces_cache_key(video_path, fps_sample)
        if not key:
            return
        cache_dir = media_root / "faces_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{key}.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fps_sample": fps_sample,
                    "faces": faces,
                    "faces_smooth": faces_smooth,
                },
                f,
                ensure_ascii=False,
            )
    except Exception:
        return None


def _estimate_frames_processed(faces: list) -> int:
    if not faces:
        return 0
    try:
        return len({f.get("time") for f in faces if "time" in f})
    except Exception:
        return len(faces)


def _get_video_frame_size(video_path: str, default: tuple[int, int] = (1920, 1080)) -> tuple[int, int]:
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if frame_w > 0 and frame_h > 0:
                return frame_w, frame_h
    except Exception:
        pass
    finally:
        if cap is not None:
            cap.release()
    return default


def _normalize_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    normalized = []
    for start, end in windows:
        if start is None or end is None:
            continue
        try:
            start_f = max(0.0, float(start))
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        if end_f <= start_f:
            continue
        normalized.append((round(start_f, 3), round(end_f, 3)))
    normalized.sort(key=lambda w: w[0])

    merged = []
    adjacency_threshold = 0.5
    for start, end in normalized:
        if not merged:
            merged.append([start, end])
            continue
        last = merged[-1]
        if start <= last[1] + adjacency_threshold:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _extract_time_windows(cut_plans: list[dict]) -> list[tuple[float, float]]:
    windows = []
    for plan in cut_plans:
        if not isinstance(plan, dict):
            continue
        windows.append((plan.get("start"), plan.get("end")))
    return _normalize_windows(windows)


def _select_fps_sample(max_window_seconds: float | None) -> int:
    if not max_window_seconds:
        return 1
    if max_window_seconds <= 60:
        return 1
    if max_window_seconds <= 300:
        return 2 if max_window_seconds <= 180 else 3
    return 4 if max_window_seconds <= 900 else 5


def _faces_cache_path_for_job(media_root: Path, job_id: int) -> Path:
    return media_root / "clips" / str(job_id) / "faces_cached.json"


def _build_faces_cache_payload(
    video_path: str,
    fps_sample: int,
    windows: list[tuple[float, float]],
    faces: list,
    faces_smooth: list,
) -> dict:
    payload = {
        "video_path": video_path,
        "fps_sample": fps_sample,
        "windows": [{"start": start, "end": end} for start, end in windows],
        "faces": faces,
        "faces_smooth": faces_smooth,
    }
    try:
        stat = os.stat(video_path)
        payload["video_mtime_ns"] = stat.st_mtime_ns
        payload["video_size"] = stat.st_size
    except OSError:
        payload["video_mtime_ns"] = None
        payload["video_size"] = None
    return payload


def _load_faces_window_cache(
    cache_path: Path,
    video_path: str,
    fps_sample: int,
    windows: list[tuple[float, float]],
) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("video_path") != video_path:
        return None
    if payload.get("fps_sample") != fps_sample:
        return None
    expected_windows = [{"start": start, "end": end} for start, end in windows]
    if payload.get("windows") != expected_windows:
        return None
    try:
        stat = os.stat(video_path)
    except OSError:
        return None
    if payload.get("video_mtime_ns") != stat.st_mtime_ns:
        return None
    if payload.get("video_size") != stat.st_size:
        return None
    return payload


def _save_faces_window_cache(cache_path: Path, payload: dict) -> None:
    try:
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return None


def _offset_faces_times(faces: list, window_start: float, window_duration: float) -> list:
    times = [f.get("time") for f in faces if isinstance(f, dict) and "time" in f]
    if not times or window_start <= 0:
        return faces
    max_time = max(times)
    if max_time <= window_duration + 0.01:
        adjusted = []
        for face in faces:
            if isinstance(face, dict) and "time" in face:
                updated = dict(face)
                updated["time"] = updated["time"] + window_start
                adjusted.append(updated)
            else:
                adjusted.append(face)
        return adjusted
    return faces


def _trim_video_window(
    video_path: str,
    start: float,
    end: float,
    temp_dir: Path,
) -> Path | None:
    try:
        duration = float(end) - float(start)
    except (TypeError, ValueError):
        return None
    if duration <= 0:
        return None
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / f"faces_window_{start:.3f}_{end:.3f}_{uuid.uuid4().hex}.mp4"
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        video_path,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        str(output_path),
    ]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    if not output_path.exists():
        return None
    return output_path


def _detect_faces_in_windows(
    video_path: str,
    windows: list[tuple[float, float]],
    fps_sample: int,
    temp_dir: Path,
) -> tuple[list, bool]:
    if not windows:
        return detect_faces(video_path, fps_sample=fps_sample), False

    faces_collected = []
    for start, end in windows:
        window_path = None
        try:
            window_path = _trim_video_window(video_path, start, end, temp_dir)
            if not window_path:
                return detect_faces(video_path, fps_sample=fps_sample), False
            window_faces = detect_faces(str(window_path), fps_sample=fps_sample)
        except Exception:
            return detect_faces(video_path, fps_sample=fps_sample), False
        finally:
            if window_path:
                try:
                    window_path.unlink()
                except OSError:
                    pass
        if window_faces:
            window_faces = _offset_faces_times(window_faces, start, end - start)
            faces_collected.extend(window_faces)
    return faces_collected, True


def _smooth_faces_linear(faces: list, alpha: float = 0.75) -> list:
    if len(faces) <= 1:
        return faces
    smoothed = []
    prev = None
    for face in faces:
        if not isinstance(face, dict):
            smoothed.append(face)
            prev = face
            continue
        if prev is None or not isinstance(prev, dict):
            smoothed_face = dict(face)
        else:
            smoothed_face = dict(face)
            if "bbox" in face and "bbox" in prev:
                try:
                    smoothed_face["bbox"] = [
                        alpha * face["bbox"][i] + (1 - alpha) * prev["bbox"][i]
                        for i in range(4)
                    ]
                except Exception:
                    smoothed_face["bbox"] = face["bbox"]
            for key in ("x", "y", "w", "h"):
                if key in face and key in prev:
                    try:
                        smoothed_face[key] = alpha * face[key] + (1 - alpha) * prev[key]
                    except Exception:
                        smoothed_face[key] = face[key]
        smoothed.append(smoothed_face)
        prev = smoothed_face
    return smoothed


def _smooth_faces_windowed(
    faces: list,
    windows: list[tuple[float, float]],
    alpha: float = 0.75,
) -> list:
    if not faces:
        return faces
    if not windows:
        return _smooth_faces_linear(faces, alpha=alpha)

    faces_sorted = sorted(
        (face for face in faces if isinstance(face, dict) and "time" in face),
        key=lambda f: f["time"],
    )
    remaining = [face for face in faces if not (isinstance(face, dict) and "time" in face)]
    windowed_faces = [[] for _ in windows]

    window_idx = 0
    for face in faces_sorted:
        t = face["time"]
        while window_idx < len(windows) and t > windows[window_idx][1]:
            window_idx += 1
        if window_idx < len(windows) and windows[window_idx][0] <= t <= windows[window_idx][1]:
            windowed_faces[window_idx].append(face)
        else:
            remaining.append(face)

    smoothed = []
    for faces_in_window in windowed_faces:
        if len(faces_in_window) <= 1:
            smoothed.extend(faces_in_window)
        else:
            smoothed.extend(_smooth_faces_linear(faces_in_window, alpha=alpha))

    if remaining:
        smoothed.extend(remaining)
    return sorted(smoothed, key=lambda f: f.get("time", 0))


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
        faces = []
        faces_smooth = []
        cache_payload = _load_faces_cache(media_root, job.original_path, fps_sample)
        if cache_payload:
            faces = cache_payload.get("faces", [])
            faces_smooth = cache_payload.get("faces_smooth", [])
            print(
                "[FACES] ✅ cache hit "
                f"fps_sample={fps_sample} frames={_estimate_frames_processed(faces)}"
            )

        if not faces:
            t0 = time.time()
            faces = detect_faces(job.original_path, fps_sample=fps_sample)
            t1 = time.time()
            print(
                "[FACES] detect_faces "
                f"fps_sample={fps_sample} "
                f"frames={_estimate_frames_processed(faces)} "
                f"t={t1 - t0:.2f}s"
            )

        if not faces_smooth:
            t0 = time.time()
            faces_smooth = smooth_faces(faces, alpha=0.75)
            t1 = time.time()
            print(
                "[FACES] smooth_faces "
                f"fps_sample={fps_sample} "
                f"frames={_estimate_frames_processed(faces)} "
                f"t={t1 - t0:.2f}s"
            )

        if faces and faces_smooth:
            _save_faces_cache(media_root, job.original_path, fps_sample, faces, faces_smooth)

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
        frame_w, frame_h = _get_video_frame_size(job.original_path)

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

        if job.processing_mode == "full":
            VideoClip.objects.get_or_create(
                job=job,
                output_path=video_path,
                defaults={
                    "start": 0,
                    "end": 0,
                    "score": 0,
                    "caption": "Vídeo completo",
                },
            )
            update_job_step(job.id, "faces", "done")
            update_job_step(job.id, "transcription", "done")
            update_job_step(job.id, "render", "done")
            update_job_step(job.id, "finalize", "done")
            job.status = "done"
            job.save(update_fields=["status"])
            return

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


def _text_story_duration(transcript: dict) -> float:
    segments = transcript.get("segments", [])
    if not segments:
        return 0.0
    return max(float(seg.get("end", 0.0)) for seg in segments)


def _write_word_srt(transcript: dict, output_path: Path) -> Path:
    duration = _text_story_duration(transcript)
    words_timeline = build_words_timeline(transcript, 0.0, duration)
    segments = [
        {"start": w["start"], "end": w["end"], "text": w["word"]}
        for w in words_timeline
    ]
    segments = fill_gaps(segments)
    srt_text = to_srt(segments)
    output_path.write_text(srt_text, encoding="utf-8")
    return output_path


@shared_task(bind=True)
def process_text_story_job(self, job_id: int, part_text: str):
    job = VideoJob.objects.get(id=job_id)
    media_root = Path(settings.MEDIA_ROOT)

    try:
        ensure_job_steps(job)

        update_job_step(job.id, "download", "running")
        job.status = "downloading"
        job.save(update_fields=["status"])
        update_job_step(job.id, "download", "done")

        update_job_step(job.id, "transcription", "running")
        job.status = "transcribing"
        job.save(update_fields=["status"])

        transcript = build_transcript_from_text(part_text)
        transcript_path = media_root / "transcripts" / f"{job.id}.json"
        transcript_path.parent.mkdir(exist_ok=True)
        transcript_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
        job.transcript_path = str(transcript_path)
        job.transcript_data = transcript
        job.save(update_fields=["transcript_path", "transcript_data"])
        update_job_step(job.id, "transcription", "done")

        update_job_step(job.id, "render", "running")
        job.status = "clipping"
        job.save(update_fields=["status"])

        duration = _text_story_duration(transcript)
        if duration <= 0:
            raise RuntimeError("Transcript duration inválida para story text")

        subs_dir = media_root / "subs"
        subs_dir.mkdir(parents=True, exist_ok=True)
        srt_path = subs_dir / f"text_story_{job.id}.srt"
        _write_word_srt(transcript, srt_path)

        bg_dir = media_root / "text_story"
        bg_path = bg_dir / f"bg_{job.id}.mp4"
        build_static_background_video(duration=duration, output_path=bg_path)

        clip = VideoClip.objects.create(
            job=job,
            start=0.0,
            end=duration,
            score=1.0,
            caption=job.title or "",
            output_path="",
        )

        out_mp4, caption = make_vertical_clip_with_captions(
            video_path=str(bg_path),
            start=0.0,
            end=duration,
            subtitle_path=str(srt_path),
            media_root=media_root,
            clip_id=str(clip.id),
        )

        clip.output_path = out_mp4
        clip.caption = caption or job.title or ""
        clip.save(update_fields=["output_path", "caption"])

        update_job_step(job.id, "render", "done")
        update_job_step(job.id, "finalize", "done")
        job.status = "done"
        job.save(update_fields=["status"])
    except Exception as e:
        fail_running_steps(job.id, message=str(e))
        job.status = "error"
        job.error = str(e)
        job.save(update_fields=["status", "error"])
        raise

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

        frame_w, frame_h = _get_video_frame_size(job.original_path)
        focus_timeline = build_focus_timeline(
            faces_tracked=faces_tracked,
            transcript=transcript,
            frame_w=frame_w,
            frame_h=frame_h,
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
        full_text = " ".join(s.get("text", "") for s in transcript.get("segments", []))
        keywords = _extract_keywords(full_text)
        archetype, _, _, _ = _classify_story(keywords, full_text)
        channel_type = "youtube_shorts"
        picks = generate_clips(
            transcript,
            channel_type=channel_type,
            archetype=archetype
        )
        # picks = pick_viral_windows_rich(transcript, min_s=18, max_s=40, top_k=6)

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
def generate_viral_clips(self, job_id: int, profile: str = "podcast"):
    """
    Gera clips virais automáticos usando window_picker + viral hooks
    """
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

        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))

        # 🔥 AQUI USA O WINDOW_PICKER COM VIRAL HOOKS

        windows = pick_viral_windows(
            transcript=transcript,
            min_s=18,
            max_s=40,
            top_n=5,
            profile=profile,  # ou job.video_profile se você adicionar no model
        )

        if not windows:
            raise ValueError("Nenhum clip viral encontrado")

        # Extrai janelas de tempo para detecção de faces
        time_windows = [(w["start"], w["end"]) for w in windows]
        max_window_seconds = max((end - start) for start, end in time_windows)

        clip_dir = media_root / "clips" / str(job.id)
        clip_dir.mkdir(parents=True, exist_ok=True)

        # DETECÇÃO DE FACES (igual você já faz)
        update_job_step(job.id, "faces", "running")
        fps_sample = _select_fps_sample(max_window_seconds)
        faces = []
        faces_smooth = []
        cache_path = _faces_cache_path_for_job(media_root, job.id)
        cache_payload = _load_faces_window_cache(
            cache_path,
            job.original_path,
            fps_sample,
            time_windows,
        )
        cache_used = False
        if cache_payload:
            faces = cache_payload.get("faces", [])
            faces_smooth = cache_payload.get("faces_smooth", [])
            cache_used = True
            print(
                "[FACES] ✅ cache hit "
                f"fps_sample={fps_sample} frames={_estimate_frames_processed(faces)}"
            )

        if not faces:
            t0 = time.time()
            windowed = False
            try:
                faces, windowed = _detect_faces_in_windows(
                    job.original_path,
                    time_windows,
                    fps_sample,
                    clip_dir / "faces_windows_tmp",
                )
            except Exception:
                faces = detect_faces(job.original_path, fps_sample=fps_sample)
            t1 = time.time()
            print(
                "[FACES] detect_faces "
                f"fps_sample={fps_sample} "
                f"frames={_estimate_frames_processed(faces)} "
                f"t={t1 - t0:.2f}s "
                f"cache={cache_used} "
                f"windowed={windowed}"
            )

        if not faces_smooth:
            t0 = time.time()
            faces_smooth = _smooth_faces_windowed(faces, time_windows, alpha=0.75)
            t1 = time.time()
            print(
                "[FACES] smooth_faces "
                f"fps_sample={fps_sample} "
                f"frames={_estimate_frames_processed(faces)} "
                f"t={t1 - t0:.2f}s "
                f"cache={cache_used}"
            )

        if faces and faces_smooth:
            payload = _build_faces_cache_payload(
                job.original_path,
                fps_sample,
                time_windows,
                faces,
                faces_smooth,
            )
            _save_faces_window_cache(cache_path, payload)

        faces_tracked = track_faces(faces_smooth) if faces_smooth else []

        faces_tracked_path = clip_dir / "faces_tracked.json"
        with open(faces_tracked_path, "w", encoding="utf-8") as f:
            json.dump(faces_tracked, f, ensure_ascii=False, indent=2)

        frame_w, frame_h = _get_video_frame_size(job.original_path)
        focus_timeline = build_focus_timeline(
            faces_tracked=faces_tracked,
            transcript=transcript,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        focus_path = clip_dir / "focus_timeline.json"
        with open(focus_path, "w", encoding="utf-8") as f:
            json.dump(focus_timeline, f, indent=2)
        update_job_step(job.id, "faces", "done")

        job.status = "clipping"
        job.save(update_fields=["status"])
        update_job_step(job.id, "render", "running")

        # 🎯 AQUI PASSA OS NOVOS PARÂMETROS anchor_start e anchor_end
        clip_tasks = []
        for win in windows:
            clip_tasks.append(
                render_clip.s(
                    job_id=job.id,
                    video_path=job.original_path,
                    transcript=transcript,
                    start=win["start"],
                    end=win["end"],
                    score=win.get("score", 0.0),
                    role=win.get("role"),
                    anchor_start=win.get("anchor_start"),  # ← NOVO
                    anchor_end=win.get("anchor_end"),  # ← NOVO
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

        cut_plans = generate_clip_sequence(transcript, blueprint_data, job.source or "other")
        if not cut_plans:
            cut_plans = [translate_blueprint_to_cut_plan(transcript, blueprint_data)]

        windows = _extract_time_windows(cut_plans)
        max_window_seconds = max((end - start) for start, end in windows) if windows else None

        clip_dir = media_root / "clips" / str(job.id)
        clip_dir.mkdir(parents=True, exist_ok=True)

        update_job_step(job.id, "faces", "running")
        fps_sample = _select_fps_sample(max_window_seconds)
        faces = []
        faces_smooth = []
        cache_path = _faces_cache_path_for_job(media_root, job.id)
        cache_payload = _load_faces_window_cache(
            cache_path,
            job.original_path,
            fps_sample,
            windows,
        )
        cache_used = False
        if cache_payload:
            faces = cache_payload.get("faces", [])
            faces_smooth = cache_payload.get("faces_smooth", [])
            cache_used = True
            print(
                "[FACES] ✅ cache hit "
                f"fps_sample={fps_sample} frames={_estimate_frames_processed(faces)}"
            )

        if not faces:
            t0 = time.time()
            windowed = False
            try:
                faces, windowed = _detect_faces_in_windows(
                    job.original_path,
                    windows,
                    fps_sample,
                    clip_dir / "faces_windows_tmp",
                )
            except Exception:
                faces = detect_faces(job.original_path, fps_sample=fps_sample)
            t1 = time.time()
            print(
                "[FACES] detect_faces "
                f"fps_sample={fps_sample} "
                f"frames={_estimate_frames_processed(faces)} "
                f"t={t1 - t0:.2f}s "
                f"cache={cache_used} "
                f"windowed={windowed}"
            )

        if not faces_smooth:
            t0 = time.time()
            faces_smooth = _smooth_faces_windowed(faces, windows, alpha=0.75)
            t1 = time.time()
            print(
                "[FACES] smooth_faces "
                f"fps_sample={fps_sample} "
                f"frames={_estimate_frames_processed(faces)} "
                f"t={t1 - t0:.2f}s "
                f"cache={cache_used}"
            )

        if faces and faces_smooth:
            payload = _build_faces_cache_payload(
                job.original_path,
                fps_sample,
                windows,
                faces,
                faces_smooth,
            )
            _save_faces_window_cache(cache_path, payload)
        faces_tracked = track_faces(faces_smooth) if faces_smooth else []

        faces_tracked_path = clip_dir / "faces_tracked.json"
        with open(faces_tracked_path, "w", encoding="utf-8") as f:
            json.dump(faces_tracked, f, ensure_ascii=False, indent=2)

        frame_w, frame_h = _get_video_frame_size(job.original_path)
        focus_timeline = build_focus_timeline(
            faces_tracked=faces_tracked,
            transcript=transcript,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        focus_path = clip_dir / "focus_timeline.json"
        with open(focus_path, "w", encoding="utf-8") as f:
            json.dump(focus_timeline, f, indent=2)
        update_job_step(job.id, "faces", "done")

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

@shared_task(bind=True)
def publish_clip_to_youtube(self, publication_id: int, channel_key: str, publish_at=None):
    publication = ClipPublication.objects.select_related("clip").get(id=publication_id)
    publication.status = "publishing"
    publication.error = ""
    publication.save(update_fields=["status", "error"])

    try:
        result = upload_clip_publication(
            publication,
            channel_key=channel_key,
            publish_at=publish_at,
        )
        publication.status = "published"
        publication.external_url = result.get("url", "")
        publication.save(update_fields=["status", "external_url"])
        return result
    except Exception as exc:
        publication.status = "error"
        publication.error = str(exc)
        publication.save(update_fields=["status", "error"])
        raise


@shared_task(bind=True)
def publish_story_clip_to_youtube(self, publication_id: int, channel_key: str, publish_at=None):
    publication = StoryClipPublication.objects.select_related("clip").get(id=publication_id)
    publication.status = "publishing"
    publication.error = ""
    publication.save(update_fields=["status", "error"])

    try:
        result = upload_story_publication(
            publication,
            channel_key=channel_key,
            publish_at=publish_at,
        )
        publication.status = "published"
        publication.external_url = result.get("url", "")
        publication.save(update_fields=["status", "external_url"])
        return result
    except Exception as exc:
        publication.status = "error"
        publication.error = str(exc)
        publication.save(update_fields=["status", "error"])
        raise


def _split_story_parts(text: str) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    matches = list(re.finditer(r"(?im)^pt\\s*\\d+\\s*[:\\-]\\s*", cleaned))
    if matches:
        parts = []
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
            part_text = cleaned[start:end].strip()
            if part_text:
                parts.append(part_text)
        return parts
    chunks = [p.strip() for p in re.split(r"\n{2,}", cleaned) if p.strip()]
    if len(chunks) > 1:
        return chunks
    sentences = re.split(r"(?<=[.!?])\\s+", cleaned)
    if len(sentences) <= 1:
        return [cleaned]
    target = max(1, math.ceil(len(sentences) / 3))
    parts = []
    for idx in range(0, len(sentences), target):
        parts.append(" ".join(sentences[idx:idx + target]).strip())
    return [p for p in parts if p]


def _build_story_words(text: str, duration_seconds: float) -> list[dict]:
    tokens = [w for w in re.split(r"\\s+", text.strip()) if w]
    if not tokens:
        return [{"start": 0.0, "end": duration_seconds, "word": "…"}]
    per_word = duration_seconds / max(len(tokens), 1)
    words = []
    cursor = 0.0
    for word in tokens:
        start = cursor
        end = min(cursor + per_word, duration_seconds)
        words.append({"start": start, "end": end, "word": word})
        cursor = end
    return words


def _estimate_story_duration(text: str) -> float:
    words = [w for w in re.split(r"\\s+", text.strip()) if w]
    if not words:
        return 8.0
    seconds = max(8.0, min(len(words) * 0.35, 30.0))
    return round(seconds, 2)


@shared_task(bind=True)
def process_story_job(self, job_id: int):
    job = VideoJob.objects.get(id=job_id)
    job.status = "clipping"
    job.save(update_fields=["status"])

    story_text = job.story_text or ""
    parts = _split_story_parts(story_text)
    if not parts:
        job.status = "error"
        job.error = "Texto da história vazio."
        job.save(update_fields=["status", "error"])
        return

    StoryClip.objects.filter(job=job).delete()
    clips = []
    for idx, part in enumerate(parts, 1):
        clips.append(StoryClip.objects.create(
            job=job,
            part_number=idx,
            text=part,
            status="pending",
        ))

    media_root = Path(settings.MEDIA_ROOT)
    subs_dir = media_root / "subs"
    subs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = media_root / "videos" / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)

    for clip in clips:
        try:
            clip.status = "processing"
            clip.save(update_fields=["status"])
            duration = _estimate_story_duration(clip.text)
            words = _build_story_words(clip.text, duration)
            ass_path = subs_dir / f"story_{job.id}_pt{clip.part_number}.ass"
            config = CaptionStyleConfig(
                font_family="Montserrat",
                font_size=44,
                font_color="#FFFFFF",
                highlight_color="#FFD84D",
                background=True,
                position=CaptionPosition.BOTTOM,
            )
            build_word_by_word_ass(words, config, str(ass_path))

            output_path = output_dir / f"story_{job.id}_pt{clip.part_number}.mp4"
            ass_filter = str(ass_path).replace("\\", "/").replace(":", "\\:")
            cmd = [
                FFMPEG_BIN, "-y",
                "-f", "lavfi",
                "-i", f"color=c=black:s=1080x1920:d={duration}",
                "-vf", f"subtitles=filename='{ass_filter}'",
                "-r", "30",
                "-pix_fmt", "yuv420p",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                str(output_path),
            ]
            subprocess.check_call(cmd)

            clip.video_path = str(output_path)
            clip.duration_seconds = duration
            clip.status = "done"
            clip.save(update_fields=["video_path", "duration_seconds", "status"])
        except Exception as e:
            clip.status = "error"
            clip.error = str(e)
            clip.save(update_fields=["status", "error"])
            job.status = "error"
            job.error = str(e)
            job.save(update_fields=["status", "error"])
            return

    job.status = "done"
    job.save(update_fields=["status"])
