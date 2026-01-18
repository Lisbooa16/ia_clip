from __future__ import annotations

from collections import defaultdict
from typing import Any


def _center_distance(face: dict[str, Any], frame_w: int, frame_h: int) -> float:
    cx = face["x"] + face["w"] / 2
    cy = face["y"] + face["h"] / 2
    dx = abs(cx - frame_w / 2) / (frame_w / 2)
    dy = abs(cy - frame_h / 2) / (frame_h / 2)
    return min(1.0, (dx + dy) / 2)


def _pick_face(
    faces_in_segment: list[dict[str, Any]],
    frame_w: int,
    frame_h: int,
) -> str | None:
    if not faces_in_segment:
        return None

    by_face: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for face in faces_in_segment:
        by_face[face["face_id"]].append(face)

    best_id = None
    best_score = -1.0

    for face_id, samples in by_face.items():
        presence = len(samples)
        avg_dist = sum(_center_distance(f, frame_w, frame_h) for f in samples) / presence
        score = presence * 2 + (1 - avg_dist)
        if score > best_score:
            best_score = score
            best_id = face_id

    return best_id


def build_focus_timeline(
    faces_tracked: list[dict[str, Any]],
    transcript: dict[str, Any],
    frame_w: int = 1920,
    frame_h: int = 1080,
    min_switch_duration: float = 0.3,
) -> list[dict[str, Any]]:
    segments = transcript.get("segments", [])
    if not segments:
        return []

    timeline: list[dict[str, Any]] = []
    last_face: str | None = None
    last_switch = 0.0

    for seg in segments:
        start = round(float(seg["start"]), 2)
        end = round(float(seg["end"]), 2)
        duration = max(end - start, 0.0)

        faces_in_segment = [
            f for f in faces_tracked
            if start <= f.get("time", 0) <= end
        ]

        chosen = _pick_face(faces_in_segment, frame_w, frame_h)

        # evita trocar muito rápido
        if chosen != last_face and duration < min_switch_duration:
            chosen = last_face

        if chosen != last_face and (start - last_switch) < min_switch_duration:
            chosen = last_face
        elif chosen != last_face:
            last_switch = start

        # 👇 NÃO zera o último rosto se perder detecção
        if chosen is None:
            chosen = last_face
        else:
            last_face = chosen

        if timeline and timeline[-1]["face_id"] == chosen:
            timeline[-1]["end"] = end
        else:
            timeline.append({
                "start": start,
                "end": end,
                "face_id": chosen,
            })

    return timeline
