# clips/tasks_clips.py
import json
import bisect
import cv2
import numpy as np
from pathlib import Path
from celery import shared_task
from django.conf import settings

from subtitles.subtitle_builder import segments_for_clip, fill_gaps, to_srt
from .domain.render_focus import (
    compute_vertical_crop,
    find_focus_face,
    focus_blocks_for_clip,
    stable_face_box,
)
from .models import VideoClip, VideoJob, ensure_job_steps, update_job_step
from .services import (
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

        transcript_segments = transcript.get("segments", [])

        # 3️⃣ SPLIT DO CLIP POR FOCO
        focus_blocks = focus_blocks_for_clip(
            focus_timeline,
            start,
            end
        )

        def _build_focus_windows():
            windows = []
            cursor = start
            for seg in transcript_segments:
                seg_start = float(seg["start"])
                seg_end = float(seg["end"])
                if seg_end <= start:
                    continue
                if seg_start >= end:
                    break
                block_start = max(seg_start, start)
                block_end = min(seg_end, end)
                if block_start > cursor:
                    windows.append({
                        "start": round(cursor, 3),
                        "end": round(block_start, 3),
                        "face_id": find_focus_face(focus_timeline, cursor, block_start),
                    })
                windows.append({
                    "start": round(block_start, 3),
                    "end": round(block_end, 3),
                    "face_id": find_focus_face(focus_timeline, block_start, block_end),
                })
                cursor = block_end
            if cursor < end:
                windows.append({
                    "start": round(cursor, 3),
                    "end": round(end, 3),
                    "face_id": find_focus_face(focus_timeline, cursor, end),
                })

            min_block = 0.2
            merged = []
            for block in windows:
                duration = block["end"] - block["start"]
                if merged:
                    prev = merged[-1]
                    if block["face_id"] == prev["face_id"]:
                        prev["end"] = max(prev["end"], block["end"])
                        continue
                    if duration < min_block:
                        prev["end"] = max(prev["end"], block["end"])
                        continue
                if duration < min_block and merged:
                    merged[-1]["end"] = max(merged[-1]["end"], block["end"])
                    continue
                merged.append(block)
            return merged

        if len(focus_blocks) <= 1 and transcript_segments and faces_tracked:
            focus_blocks = _build_focus_windows() or focus_blocks
            print(
                "[RENDER] 🧭 "
                f"rewindow blocks={len(focus_blocks)} clip_id={clip.id}"
            )

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
            face_box = stable_face_box(
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
            print(
                "[FOCUS] 🎥 "
                f"virtual_camera face_id={face_id} crop=({crop['x']},{crop['y']})"
            )
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
            print("[FOCUS] 🎥 virtual_camera center")
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
        else:
            print(f"[FOCUS] 🎥 virtual_cameras_ready={sorted(virtual_cameras)}")

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
        min_motion_window = 0.2
        transcript_boost = 1.5
        switch_hysteresis = 0.2

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

        def _dominant_face_id():
            counts = {}
            for face in faces_tracked:
                t = face.get("time", 0.0)
                face_id = face.get("face_id")
                if face_id is None or t < start or t > end:
                    continue
                counts[face_id] = counts.get(face_id, 0) + 1
            if not counts:
                return None
            return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

        dominant_face_id = _dominant_face_id()
        if len(visible_face_ids) == 1:
            base_blocks = [{
                "start": round(start, 3),
                "end": round(end, 3),
                "face_id": visible_face_ids[0],
            }]
        elif focus_blocks:
            base_blocks = focus_blocks
        else:
            base_blocks = [{
                "start": round(start, 3),
                "end": round(end, 3),
                "face_id": dominant_face_id,
            }]

        speaker_timeline = []
        last_face_id = None
        last_switch_time = start
        for idx, block in enumerate(base_blocks):
            block_start = block["start"]
            block_end = block["end"]
            if block_end <= block_start:
                continue

            block_visible = sorted({
                face.get("face_id")
                for face in faces_tracked
                if block_start <= face.get("time", 0) <= block_end
                and face.get("face_id") is not None
            })
            block_visible = [fid for fid in block_visible if fid is not None]
            print(
                "[FOCUS] 👁️ "
                f"{block_start:.3f}-{block_end:.3f}s visible={block_visible}"
            )

            has_speech = any(
                seg["end"] > block_start and seg["start"] < block_end
                for seg in transcript_segments
            )
            candidate_id = block.get("face_id")
            if candidate_id not in block_visible:
                candidate_id = None
            if candidate_id is None and block_visible:
                candidate_id = block_visible[0]

            selected_face_id = candidate_id or last_face_id
            selection_reason = "timeline"
            motion_scores = {}
            if use_motion_check and block_visible:
                score_start = block_start + confirm_offset
                score_end = min(block_end, score_start + confirm_window)
                window_duration = score_end - score_start
                if window_duration >= min_motion_window:
                    for candidate in block_visible:
                        motion_scores[candidate] = _mouth_motion_score(
                            candidate,
                            score_start,
                            score_end,
                        ) or 0.0
                    active_scores = {
                        fid: score
                        for fid, score in motion_scores.items()
                        if score >= motion_threshold
                    }
                    if has_speech and candidate_id in active_scores:
                        active_scores[candidate_id] = (
                            active_scores[candidate_id] + transcript_boost
                        )
                    print(
                        f"[FOCUS] 👥 visible={block_visible} "
                        f"scores={motion_scores} active={active_scores}"
                    )
                    if active_scores:
                        if candidate_id in active_scores:
                            selected_face_id = candidate_id
                            selection_reason = "transcript_confirmed"
                        elif last_face_id in active_scores:
                            selected_face_id = last_face_id
                            selection_reason = "hold_last_active"
                        elif len(active_scores) == 1:
                            selected_face_id = next(iter(active_scores))
                            selection_reason = "motion_single"
                        else:
                            selected_face_id = last_face_id or candidate_id
                            selection_reason = "motion_ambiguous"
                    else:
                        selected_face_id = last_face_id or candidate_id
                        selection_reason = "no_active"
                else:
                    selected_face_id = last_face_id or candidate_id
                    selection_reason = "short_window"

            if not block_visible:
                selected_face_id = last_face_id
                selection_reason = "no_faces_hold"

            if selected_face_id is None and dominant_face_id is not None:
                selected_face_id = dominant_face_id
                selection_reason = "dominant_fallback"

            if selected_face_id != last_face_id and last_face_id is not None:
                if (block_start - last_switch_time) < switch_hysteresis:
                    selected_face_id = last_face_id
                    selection_reason = "hysteresis_hold"
                else:
                    print(
                        "[FOCUS] 🎬 "
                        f"camera_switch {last_face_id}->{selected_face_id} "
                        f"t={block_start:.3f}s"
                    )
                    last_switch_time = block_start

            if speaker_timeline and speaker_timeline[-1]["face_id"] == selected_face_id:
                speaker_timeline[-1]["end"] = block_end
            else:
                speaker_timeline.append({
                    "start": round(block_start, 3),
                    "end": round(block_end, 3),
                    "face_id": selected_face_id,
                    "reason": selection_reason,
                })

            print(
                "[FOCUS] ✅ "
                f"{block_start:.3f}-{block_end:.3f}s "
                f"speaker={selected_face_id} reason={selection_reason}"
            )

            if selected_face_id is not None:
                last_face_id = selected_face_id

        print("[FOCUS] 🧾 speaker_timeline:")
        for entry in speaker_timeline:
            print(
                "[FOCUS] 🧾 "
                f"{entry['start']:.3f}-{entry['end']:.3f}s "
                f"face_id={entry['face_id']} reason={entry['reason']}"
            )

        for idx, block in enumerate(speaker_timeline):
            block_start = block["start"]
            block_end = block["end"]
            if block_end <= block_start:
                continue
            face_id = block.get("face_id")
            cam_path = None
            if face_id in virtual_cameras:
                cam_path = virtual_cameras[face_id]
            elif center_cam:
                cam_path = center_cam
            elif virtual_cameras:
                cam_path = next(iter(virtual_cameras.values()))

            if cam_path is None:
                continue

            rel_start = block_start - start
            rel_end = block_end - start
            if rel_end <= rel_start:
                continue

            temp_out = media_root / "tmp" / f"{clip.id}_{idx}.mp4"
            temp_out.parent.mkdir(parents=True, exist_ok=True)
            print(
                "[RENDER] 🧩 "
                f"segment={idx} {block_start:.3f}-{block_end:.3f}s "
                f"camera={face_id}"
            )
            trim_clip(
                video_path=str(cam_path),
                start=rel_start,
                end=rel_end,
                output_path=temp_out,
            )
            temp_files.append(temp_out)

        if cap:
            cap.release()

        if not temp_files:
            fallback_cam = center_cam or next(iter(virtual_cameras.values()), None)
            if fallback_cam:
                temp_out = media_root / "tmp" / f"{clip.id}_fallback.mp4"
                temp_out.parent.mkdir(parents=True, exist_ok=True)
                trim_clip(
                    video_path=str(fallback_cam),
                    start=0.0,
                    end=end - start,
                    output_path=temp_out,
                )
                temp_files.append(temp_out)

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
