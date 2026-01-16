# clips/tasks_clips.py
import json
import bisect
import cv2
import numpy as np
from pathlib import Path
from celery import shared_task
from django.conf import settings

from subtitles.subtitle_builder import segments_for_clip, fill_gaps, to_srt
from .domain.render_focus import average_face_box, compute_vertical_crop, find_focus_face, focus_blocks_for_clip
from .models import VideoClip, VideoJob, ensure_job_steps, update_job_step
from .services import (
    make_vertical_clip_with_captions,
    make_vertical_clip_with_focus,
    trim_clip,
    burn_subtitles,
    FFMPEG_BIN,
)
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
        virtual_cameras = {}
        visible_face_ids = sorted({
            face.get("face_id")
            for face in faces_tracked
            if start <= face.get("time", 0) <= end
            and face.get("face_id") is not None
        })
        print(f"[FOCUS] 🎥 faces_in_clip={visible_face_ids}")
        for face_id in visible_face_ids:
            face_box = average_face_box(
                faces_tracked,
                face_id,
                start,
                end,
            )
            if not face_box:
                continue
            crop = compute_vertical_crop(
                face_box,
                frame_w=1920,
                frame_h=1080,
            )
            cam_path = media_root / "tmp" / f"{clip.id}_cam_{face_id}.mp4"
            cam_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[FOCUS] 🎥 render camera face_id={face_id} crop_x={crop['x']}")
            make_vertical_clip_with_focus(
                video_path=video_path,
                start=start,
                end=end,
                subtitle_path=None,
                media_root=media_root,
                clip_id=cam_path.stem,
                crop=crop,
                output_path=cam_path,
            )
            virtual_cameras[face_id] = cam_path

        center_cam = None
        if not virtual_cameras:
            center_cam = media_root / "tmp" / f"{clip.id}_cam_center.mp4"
            center_cam.parent.mkdir(parents=True, exist_ok=True)
            print("[FOCUS] 🎥 render camera center")
            make_vertical_clip_with_focus(
                video_path=video_path,
                start=start,
                end=end,
                subtitle_path=None,
                media_root=media_root,
                clip_id=center_cam.stem,
                crop=None,
                output_path=center_cam,
            )

        face_index: dict[str, list[tuple[float, dict]]] = {}
        for face in faces_tracked:
            face_id = face.get("face_id")
            if face_id is None:
                continue
            face_index.setdefault(face_id, []).append((face.get("time", 0.0), face))
        for face_id in face_index:
            face_index[face_id].sort(key=lambda item: item[0])

        use_motion_check = len(face_index) > 0
        cap = cv2.VideoCapture(video_path) if use_motion_check else None
        if cap and not cap.isOpened():
            cap.release()
            cap = None
            use_motion_check = False
        motion_threshold = 6.0
        confirm_offset = 0.0
        confirm_window = 0.25
        silence_hold = 0.4
        min_motion_window = 0.2
        transcript_boost = 1.5

        def _find_face_box(face_id, t):
            samples = face_index.get(face_id)
            if not samples:
                return None
            times = [item[0] for item in samples]
            idx = bisect.bisect_left(times, t)
            if idx >= len(samples):
                idx = len(samples) - 1
            return samples[idx][1]

        def _mouth_region(face_box):
            x = int(face_box["x"] + 0.2 * face_box["w"])
            y = int(face_box["y"] + 0.6 * face_box["h"])
            w = int(face_box["w"] * 0.6)
            h = int(face_box["h"] * 0.3)
            return x, y, w, h

        def _mouth_motion_score(face_id, start_t, end_t):
            if not cap or start_t >= end_t:
                return None
            step = 1 / 6
            times = np.arange(start_t, end_t, step)
            prev = None
            total = 0.0
            count = 0
            logged_resize = False
            logged_skip = False
            logged_shape = False
            for t in times:
                face_box = _find_face_box(face_id, t)
                if not face_box:
                    if not logged_skip:
                        print(f"[MOUTH] ⏭️ skip no face box t={t:.2f}")
                        logged_skip = True
                    total += 0.0
                    count += 1
                    prev = None
                    continue
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if not ok:
                    if not logged_skip:
                        print(f"[MOUTH] ⏭️ skip frame read t={t:.2f}")
                        logged_skip = True
                    total += 0.0
                    count += 1
                    prev = None
                    continue
                x, y, w, h = _mouth_region(face_box)
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    if not logged_skip:
                        print(f"[MOUTH] ⏭️ skip invalid roi t={t:.2f} roi=({x},{y},{w},{h})")
                        logged_skip = True
                    total += 0.0
                    count += 1
                    prev = None
                    continue
                if x + w > frame.shape[1] or y + h > frame.shape[0]:
                    if not logged_skip:
                        print(f"[MOUTH] ⏭️ skip oob roi t={t:.2f} roi=({x},{y},{w},{h})")
                        logged_skip = True
                    total += 0.0
                    count += 1
                    prev = None
                    continue

                mouth = frame[y:y + h, x:x + w]
                if mouth.size == 0:
                    if not logged_skip:
                        print(f"[MOUTH] ⏭️ skip empty roi t={t:.2f}")
                        logged_skip = True
                    total += 0.0
                    count += 1
                    prev = None
                    continue

                try:
                    gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
                    target_size = (64, 32)
                    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
                    if not logged_resize:
                        print(f"[MOUTH] 📏 resize roi to={target_size}")
                        logged_resize = True
                    if prev is not None:
                        if not logged_shape:
                            print(f"[MOUTH] 🔍 diff roi shape prev={prev.shape} curr={resized.shape}")
                            logged_shape = True
                        diff = cv2.absdiff(prev, resized)
                        total += float(np.mean(diff))
                        count += 1
                    prev = resized
                except cv2.error:
                    print("[MOUTH] ⚠️ cv2.error diff fallback=0")
                    total += 0.0
                    count += 1
                    prev = None
            if count == 0:
                return None
            return total / count

        transcript_segments = transcript.get("segments", [])
        last_speech_end = start
        current_silence = 0.0
        # 4️⃣ RENDER DE CADA BLOCO
        last_face_id = None
        last_camera_id = None
        for idx, block in enumerate(focus_blocks):
            face_id = block["face_id"]
            block_start = block["start"]
            block_end = block["end"]
            block_duration = block_end - block_start

            visible_face_ids = sorted({
                face.get("face_id")
                for face in faces_tracked
                if block_start <= face.get("time", 0) <= block_end
                and face.get("face_id") is not None
            })
            visible_face_ids = [fid for fid in visible_face_ids if fid is not None]
            print(
                "[FOCUS] 👁️ "
                f"{block_start:.3f}-{block_end:.3f}s visible={visible_face_ids}"
            )

            has_speech = any(
                seg["end"] > block_start and seg["start"] < block_end
                for seg in transcript_segments
            )
            selected_face_id = face_id
            selection_reason = "timeline"
            motion_scores = {}
            if use_motion_check and visible_face_ids:
                score_start = block_start + confirm_offset
                score_end = min(block_end, score_start + confirm_window)
                window_duration = score_end - score_start
                if window_duration >= min_motion_window:
                    for candidate_id in visible_face_ids:
                        motion_scores[candidate_id] = _mouth_motion_score(
                            candidate_id,
                            score_start,
                            score_end,
                        ) or 0.0
                    active_scores = {
                        fid: score
                        for fid, score in motion_scores.items()
                        if score >= motion_threshold
                    }
                    boosted_scores = dict(active_scores)
                    if has_speech and face_id in boosted_scores:
                        boosted_scores[face_id] = boosted_scores[face_id] + transcript_boost
                    print(
                        f"[FOCUS] 👥 visible={visible_face_ids} "
                        f"scores={motion_scores} active={active_scores} boosted={boosted_scores}"
                    )
                    best_id = None
                    best_score = None
                    if boosted_scores:
                        best_id = max(boosted_scores, key=boosted_scores.get)
                        best_score = boosted_scores[best_id]

                    current_focus_id = last_face_id or face_id
                    current_motion = motion_scores.get(current_focus_id, 0.0)
                    if current_focus_id is None:
                        current_silence = 0.0
                    elif current_motion >= motion_threshold:
                        current_silence = 0.0
                    else:
                        current_silence += block_duration

                    if best_id is None:
                        selection_reason = "no_active"
                        selected_face_id = current_focus_id or face_id
                    elif current_focus_id is None:
                        selection_reason = "initial_motion"
                        selected_face_id = best_id
                    elif current_motion >= motion_threshold:
                        selection_reason = "current_motion"
                        selected_face_id = current_focus_id
                    elif current_silence < silence_hold:
                        selection_reason = "hold_silence"
                        selected_face_id = current_focus_id
                    else:
                        selection_reason = "switch_motion"
                        selected_face_id = best_id

                    print(
                        "[FOCUS] 🎯 "
                        f"select={selected_face_id} reason={selection_reason} "
                        f"best={best_id} best_score={best_score} silence={current_silence:.2f}"
                    )
                else:
                    print(
                        "[FOCUS] ⏳ "
                        f"skip motion window={window_duration:.2f}s"
                    )
                    selected_face_id = last_face_id or face_id
                    selection_reason = "short_window"

            if selected_face_id != face_id:
                print(
                    "[FOCUS] 🔁 "
                    f"override {face_id}->{selected_face_id} "
                    f"speech={has_speech} reason={selection_reason}"
                )
                face_id = selected_face_id
            print(
                "[FOCUS] ✅ "
                f"selected face_id={face_id} reason={selection_reason} speech={has_speech}"
            )

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

            if face_id is None and last_face_id is not None and (block_start - last_speech_end) < silence_hold:
                face_id = last_face_id
                crop = last_crop
            elif face_id is None:
                crop = None
            elif crop is None and last_crop is not None:
                crop = last_crop
            elif last_face_id is not None and face_id == last_face_id and last_crop is not None:
                crop = last_crop

            if crop:
                print(
                    "[RENDER] ✂️ "
                    f"{block['start']:.3f}-{block['end']:.3f}s "
                    f"crop x={crop['x']} y={crop['y']} w={crop['w']} h={crop['h']}"
                )
            else:
                print(
                    "[RENDER] ✂️ "
                    f"{block['start']:.3f}-{block['end']:.3f}s center"
                )

            requested_switch = (
                last_crop
                and crop
                and last_face_id is not None
                and face_id is not None
                and face_id != last_face_id
            )
            confirmed_switch = requested_switch

            if requested_switch and confirmed_switch and block_duration > min_transition:
                pre_focus_end = min(block_start + confirm_window, block_end)
                if last_crop and pre_focus_end > block_start:
                    temp_out = media_root / "tmp" / f"{clip.id}_{idx}_pre.mp4"
                    temp_out.parent.mkdir(parents=True, exist_ok=True)
                    make_vertical_clip_with_focus(
                        video_path=video_path,
                        start=block_start,
                        end=pre_focus_end,
                        subtitle_path=str(srt_path),
                        media_root=media_root,
                        clip_id=temp_out.stem,
                        crop=last_crop,
                        output_path=temp_out,
                    )
                    temp_files.append(temp_out)
                    block_start = pre_focus_end

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

            if face_id is None and last_face_id is not None and (block_start - last_speech_end) < silence_hold:
                face_id = last_face_id

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

            if face_id is not None:
                last_face_id = face_id
                last_camera_id = face_id
                last_speech_end = block_end

        if cap:
            cap.release()

        # 5️⃣ CONCATENA
        concat_file = media_root / "tmp" / f"{clip.id}_concat.txt"
        with open(concat_file, "w") as f:
            for t in temp_files:
                f.write(f"file '{t.as_posix()}'\n")

        final_out = media_root / "videos" / "clips" / f"{clip.id}.mp4"
        final_out.parent.mkdir(parents=True, exist_ok=True)
        concat_out = media_root / "tmp" / f"{clip.id}_concat.mp4"

        cmd = [
            FFMPEG_BIN, "-y",
            "-f", "concat",
            "-safe", "0",
            "-fflags", "+genpts",
            "-i", str(concat_file),
            "-r", "30",
            "-fps_mode", "cfr",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-af", "aresample=async=1:first_pts=0",
            "-movflags", "+faststart",
            str(concat_out),
        ]
        print("[RENDER] 🎞️ concat_fps=30 res=1080x1920 pix_fmt=yuv420p")
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            fallback_cmd = cmd[:]
            fallback_cmd[fallback_cmd.index("-preset") + 1] = "ultrafast"
            fallback_cmd[fallback_cmd.index("-crf") + 1] = "23"
            print("[RENDER] ⚠️ concat fallback retry")
            subprocess.check_call(fallback_cmd)

        if srt_path.exists():
            print("[SUB] 🎬 burn subtitles on final output")
            burn_subtitles(
                video_path=str(concat_out),
                subtitle_path=str(srt_path),
                output_path=final_out,
            )
        else:
            concat_out.replace(final_out)

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
