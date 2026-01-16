# clips/tasks_clips.py
import json
import uuid
from pathlib import Path
from celery import shared_task
from django.conf import settings

from subtitles.subtitle_builder import segments_for_clip, fill_gaps, to_srt
from .domain.render_focus import average_face_box, compute_vertical_crop, find_focus_face, focus_blocks_for_clip
from .models import VideoClip, VideoJob, ensure_job_steps, update_job_step
from .services import make_vertical_clip_with_captions, make_vertical_clip_with_focus, FFMPEG_BIN
import subprocess


@shared_task(bind=True)
def render_clip(
    self,
    job_id: int,
    video_path: str,
    transcript: dict,
    start: float,
    end: float,
    score: float,
    role: str | None = None,
):
    media_root = Path(settings.MEDIA_ROOT)
    clip_dir = media_root / "clips" / str(job_id)
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip = VideoClip.objects.create(
        job_id=job_id,
        start=start,
        end=end,
        score=score,
        caption=role or "",
        output_path="",
    )

    try:
        # 1️⃣ LEGENDA (igual você já faz)
        segments = segments_for_clip(transcript["segments"], start, end)
        segments = fill_gaps(segments)
        srt_text = to_srt(segments)

        subs_dir = media_root / "subs"
        subs_dir.mkdir(parents=True, exist_ok=True)
        srt_path = subs_dir / f"{clip.id}.srt"
        srt_path.write_text(srt_text, encoding="utf-8")

        # 2️⃣ CARREGA FOCO E FACES
        focus_path = clip_dir / "focus_timeline.json"
        if not focus_path.exists():
            raise RuntimeError("focus_timeline.json não encontrado")

        with open(focus_path) as f:
            focus_timeline = json.load(f)

        faces_path = clip_dir / "faces_tracked.json"
        if faces_path.exists():
            with open(faces_path) as f:
                faces_tracked = json.load(f)
        else:
            faces_tracked = []

        # 3️⃣ SPLIT DO CLIP POR FOCO
        focus_blocks = focus_blocks_for_clip(
            focus_timeline,
            start,
            end
        )

        if not focus_blocks:
            out_mp4, caption = make_vertical_clip_with_captions(
                video_path=video_path,
                start=start,
                end=end,
                subtitle_path=str(srt_path),
                media_root=media_root,
                clip_id=str(clip.id),
            )
            clip.output_path = out_mp4
            clip.caption = caption
            clip.save(update_fields=["output_path", "caption"])
            return str(clip.id)

        print(f"[RENDER] 🎯 focus_blocks={len(focus_blocks)} clip_id={clip.id}")

        temp_files = []
        last_crop = None
        transition_steps = 4
        min_transition = 0.2
        max_transition = 0.4

        def interpolate_crop(a, b, alpha):
            if not a or not b:
                return b or a
            eased = alpha * alpha * (3 - 2 * alpha)
            return {
                "x": int(round(a["x"] + (b["x"] - a["x"]) * eased)),
                "y": int(round(a["y"] + (b["y"] - a["y"]) * eased)),
                "w": a["w"],
                "h": a["h"],
            }

        # 4️⃣ RENDER DE CADA BLOCO
        last_face_id = None
        for idx, block in enumerate(focus_blocks):
            face_id = block["face_id"]

            face_box = None
            if face_id is not None:
                face_box = average_face_box(
                    faces_tracked,
                    face_id,
                    block["start"],
                    block["end"],
                )

            crop = None
            if face_box:
                crop = compute_vertical_crop(
                    face_box,
                    frame_w=1920,
                    frame_h=1080,
                )

            if face_id is None:
                crop = None
            elif crop is None and last_crop is not None:
                crop = last_crop
            elif last_face_id is not None and face_id == last_face_id and last_crop is not None:
                crop = last_crop

            if crop:
                print(
                    "[RENDER] ✂️ "
                    f"{block['start']:.3f}-{block['end']:.3f}s "
                    f"crop x={crop['x']} w={crop['w']}"
                )
            else:
                print(
                    "[RENDER] ✂️ "
                    f"{block['start']:.3f}-{block['end']:.3f}s center"
                )

            block_start = block["start"]
            block_end = block["end"]
            block_duration = block_end - block_start

            focus_changed = (
                last_crop
                and crop
                and last_face_id is not None
                and face_id is not None
                and face_id != last_face_id
            )
            if focus_changed and block_duration > min_transition:
                transition_dur = min(max_transition, max(min_transition, block_duration / 2))
                if transition_dur > block_duration:
                    transition_dur = block_duration
                step_dur = transition_dur / transition_steps

                for step in range(transition_steps):
                    seg_start = block_start + step * step_dur
                    seg_end = seg_start + step_dur
                    alpha = (step + 1) / transition_steps
                    step_crop = interpolate_crop(last_crop, crop, alpha)
                    print(
                        "[RENDER] 🎞️ "
                        f"{seg_start:.3f}-{seg_end:.3f}s "
                        f"crop x={step_crop['x']}"
                    )

                    temp_out = media_root / "tmp" / f"{clip.id}_{idx}_t{step}.mp4"
                    temp_out.parent.mkdir(parents=True, exist_ok=True)

                    make_vertical_clip_with_focus(
                        video_path=video_path,
                        start=seg_start,
                        end=seg_end,
                        subtitle_path=str(srt_path),
                        media_root=media_root,
                        clip_id=temp_out.stem,
                        crop=step_crop,
                        output_path=temp_out,
                    )
                    temp_files.append(temp_out)

                block_start += transition_dur

            if block_end - block_start > 0.001:
                temp_out = media_root / "tmp" / f"{clip.id}_{idx}.mp4"
                temp_out.parent.mkdir(parents=True, exist_ok=True)

                make_vertical_clip_with_focus(
                    video_path=video_path,
                    start=block_start,
                    end=block_end,
                    subtitle_path=str(srt_path),
                    media_root=media_root,
                    clip_id=temp_out.stem,
                    crop=crop,
                    output_path=temp_out,
                )

                temp_files.append(temp_out)

            if crop is not None:
                last_crop = crop
            if face_id is not None:
                last_face_id = face_id

        # 5️⃣ CONCATENA
        concat_file = media_root / "tmp" / f"{clip.id}_concat.txt"
        with open(concat_file, "w") as f:
            for t in temp_files:
                f.write(f"file '{t.as_posix()}'\n")

        final_out = media_root / "videos" / "clips" / f"{clip.id}.mp4"
        final_out.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            FFMPEG_BIN, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(final_out),
        ]

        subprocess.check_call(cmd)

        clip.output_path = str(final_out)
        clip.save(update_fields=["output_path"])

    except Exception as e:
        update_job_step(job_id, "render", "failed", message=str(e))
        clip.caption = "Erro ao renderizar clip"
        clip.save(update_fields=["caption"])
        raise

    return str(clip.id)


@shared_task(bind=True)
def finalize_job(self, _results, job_id: int):
    job = VideoJob.objects.get(id=job_id)
    ensure_job_steps(job)
    update_job_step(job.id, "render", "done")
    update_job_step(job.id, "finalize", "running")
    job.status = "done"
    job.save(update_fields=["status"])
    update_job_step(job.id, "finalize", "done")
