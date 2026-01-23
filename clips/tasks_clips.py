import json
import bisect
import cv2
import numpy as np
from pathlib import Path
from celery import shared_task
from django.conf import settings

from subtitles.subtitle_builder import build_subtitle_artifacts
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
    anchor_start: float | None = None,  # ← NOVO parâmetro (pico de hook)
    anchor_end: float | None = None,    # ← NOVO parâmetro (fim do hook)
):
    media_root = Path(settings.MEDIA_ROOT)
    clip_dir = media_root / "clips" / str(job_id)
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip = VideoClip.objects.create(
        job_id=job_id,
        start=start,
        end=end,
        original_start=start,
        original_end=end,
        original_video_path=video_path,
        score=score,
        caption=role or "",
        output_path="",
    )

    try:
        # 1️⃣ LEGENDA (como já fazia)
        subs_dir = media_root / "subs"
        srt_path, subtitle_style, subtitle_config = build_subtitle_artifacts(
            transcript=transcript,
            clip_start=start,
            clip_end=end,
            caption_style=clip.caption_style,
            caption_config=clip.caption_config,
            output_dir=subs_dir,
            clip_id=str(clip.id),
        )

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

        # 3️⃣ GERA FOCUS BLOCKS PARA O CLIP
        focus_blocks = focus_blocks_for_clip(
            focus_timeline,
            start,
            end
        )

        # Se não tiver focus blocks, mantém comportamento antigo (sem hook replay)
        if not focus_blocks:
            out_mp4, caption = make_vertical_clip_with_captions(
                video_path=video_path,
                start=start,
                end=end,
                subtitle_path=str(srt_path),
                media_root=media_root,
                clip_id=str(clip.id),
                caption_style=subtitle_style,
                caption_config=subtitle_config.__dict__,
            )
            clip.output_path = out_mp4
            clip.caption = caption
            clip.save(update_fields=["output_path", "caption"])
            return str(clip.id)

        print(f"[RENDER] 🎯 focus_blocks={len(focus_blocks)} clip_id={clip.id}")

        temp_files = []

        # 3.1️⃣ HOOK REPLAY (opcional)
        # Se vier anchor_start/anchor_end, criamos um mini preview antes do clipe completo.
        if anchor_start is not None:
            HOOK_PREVIEW_MAX_DURATION = 6.0   # duração máx. do replay
            HOOK_PREVIEW_LEAD_IN = 0.5        # segundos antes do anchor_start
            HOOK_PREVIEW_TAIL = 0.5           # segundos depois do anchor_end/anchor_start
            HOOK_MIN_LEAD_FROM_START = 1.0    # evita repetir o começo do clip

            base = anchor_start
            anchor_tail = anchor_end if anchor_end is not None else anchor_start

            preview_start = max(start, base - HOOK_PREVIEW_LEAD_IN)
            preview_end = min(end, anchor_tail + HOOK_PREVIEW_TAIL)

            if preview_end - preview_start > HOOK_PREVIEW_MAX_DURATION:
                preview_end = preview_start + HOOK_PREVIEW_MAX_DURATION

            if preview_end - preview_start >= 0.7 and (preview_start - start) >= HOOK_MIN_LEAD_FROM_START:
                hook_out = media_root / "tmp" / f"{clip.id}_hook.mp4"
                hook_out.parent.mkdir(parents=True, exist_ok=True)

                print(
                    "[HOOK] 🎬 preview "
                    f"{preview_start:.3f}-{preview_end:.3f}s "
                    f"(anchor={anchor_start:.3f}-{(anchor_end or anchor_start):.3f}s)"
                )

                # Aqui usamos legenda + corte simples (sem lógica de foco)
                hook_srt_path, hook_style, hook_config = build_subtitle_artifacts(
                    transcript=transcript,
                    clip_start=preview_start,
                    clip_end=preview_end,
                    caption_style=clip.caption_style,
                    caption_config=clip.caption_config,
                    output_dir=subs_dir,
                    clip_id=str(clip.id),
                    suffix="_hook",
                )
                make_vertical_clip_with_captions(
                    video_path=video_path,
                    start=preview_start,
                    end=preview_end,
                    subtitle_path=str(hook_srt_path),
                    media_root=media_root,
                    clip_id=f"{clip.id}_hook",
                    output_path=hook_out,
                    caption_style=hook_style,
                    caption_config=hook_config.__dict__,
                )
                temp_files.append(hook_out)

        # 4️⃣ RENDER POR BLOCO DE FOCO (lógica de speaker focus + mouth motion)
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

        # Hiperparâmetros de foco de fala
        motion_threshold = 2.5
        motion_delta = 0.8
        silence_hold = 0.5
        confirm_offset = 0.0
        confirm_window = 0.25
        min_motion_window = 0.15
        transcript_boost = 1.0

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
        last_face_id = None

        for idx, block in enumerate(focus_blocks):
            face_id = block["face_id"]
            block_start = block["start"]
            block_end = block["end"]
            block_duration = block_end - block_start

            visible_face_ids = sorted({
                face.get("face_id")
                for face in faces_tracked
                if block_start <= face.get("time", 0) <= block_end
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

                if window_duration >= min_motion_window and has_speech:
                    # ⚡ Só analisamos boca se há fala nesse bloco
                    for candidate_id in visible_face_ids:
                        motion_scores[candidate_id] = _mouth_motion_score(
                            candidate_id,
                            score_start,
                            score_end,
                        ) or 0.0

                    current_focus_id = last_face_id or face_id
                    current_motion = motion_scores.get(current_focus_id, 0.0)

                    # melhor candidato por movimento de boca
                    if motion_scores:
                        best_id = max(motion_scores, key=motion_scores.get)
                        best_score = motion_scores[best_id]
                    else:
                        best_id = None
                        best_score = 0.0

                    # pequeno viés para o rosto “previsto” na timeline
                    if face_id is not None and face_id in motion_scores:
                        motion_scores[face_id] = motion_scores[face_id] + transcript_boost
                        best_id = max(motion_scores, key=motion_scores.get)
                        best_score = motion_scores[best_id]

                    # Ninguém mexendo a boca o suficiente → segura último foco
                    if best_score < motion_threshold:
                        selected_face_id = current_focus_id or best_id or face_id
                        selection_reason = "below_threshold_hold"
                        current_silence += block_duration
                    else:
                        # ainda não temos foco definido → assume melhor candidato
                        if current_focus_id is None:
                            selected_face_id = best_id
                            selection_reason = "initial_speaker"
                            current_silence = 0.0
                        else:
                            # mesmo rosto continua falando
                            if best_id == current_focus_id:
                                selected_face_id = current_focus_id
                                selection_reason = "same_speaker"
                                current_silence = 0.0
                            else:
                                # novo rosto com boca forte:
                                # troca só se ele é bem melhor OU já estamos em silêncio
                                if (
                                    best_score >= current_motion + motion_delta
                                    or current_silence >= silence_hold
                                ):
                                    selected_face_id = best_id
                                    selection_reason = "switch_speaker"
                                    current_silence = 0.0
                                else:
                                    selected_face_id = current_focus_id
                                    selection_reason = "hold_current"
                                    current_silence += block_duration

                    print(
                        "[FOCUS] 👥 visible=%s scores=%s best=%s best_score=%.2f "
                        "current=%s current_motion=%.2f reason=%s silence=%.2f"
                        % (
                            visible_face_ids,
                            motion_scores,
                            best_id,
                            best_score,
                            current_focus_id,
                            current_motion,
                            selection_reason,
                            current_silence,
                        )
                    )
                else:
                    # ❌ Sem fala ou janela muito curta → NÃO troca foco
                    selected_face_id = last_face_id or face_id
                    selection_reason = "short_or_no_speech"

            if selected_face_id != face_id:
                print(
                    "[FOCUS] 🔁 override %s->%s speech=%s reason=%s"
                    % (face_id, selected_face_id, has_speech, selection_reason)
                )
            face_id = selected_face_id

            print(
                "[FOCUS] ✅ selected face_id=%s reason=%s speech=%s"
                % (face_id, selection_reason, has_speech)
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

            # fallback de foco em silêncio curto
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

            # TRANSIÇÃO SUAVE ENTRE ROSTOS
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
                        caption_style=subtitle_style,
                        caption_config=subtitle_config.__dict__,
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
                        caption_style=subtitle_style,
                        caption_config=subtitle_config.__dict__,
                    )
                    temp_files.append(temp_out)

                block_start += transition_dur

            if not confirmed_switch and requested_switch:
                face_id = last_face_id
                crop = last_crop

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
                    caption_style=subtitle_style,
                    caption_config=subtitle_config.__dict__,
                )

                temp_files.append(temp_out)

            if crop is not None:
                last_crop = crop
            if face_id is not None:
                last_face_id = face_id
                last_speech_end = block_end

        if cap:
            cap.release()

        # 5️⃣ CONCATENA (Hook Replay + blocos de foco)
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
            str(final_out),
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
