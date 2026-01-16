from __future__ import annotations

from typing import Any


def _segment_word_count(segment: dict[str, Any]) -> int:
    words = segment.get("words") or []
    if words:
        return len(words)
    return len((segment.get("text") or "").split())


def _window_density(segments: list[dict[str, Any]], start: float, end: float) -> float:
    word_count = 0
    for seg in segments:
        if seg["end"] <= start:
            continue
        if seg["start"] >= end:
            break
        word_count += _segment_word_count(seg)
    duration = max(end - start, 0.1)
    return word_count / duration


def _best_window(
    segments: list[dict[str, Any]],
    total_duration: float,
    target_length: float,
    start_min: float,
    start_max: float,
) -> tuple[float, float]:
    best = (0.0, min(target_length, total_duration))
    best_score = -1.0

    for seg in segments:
        start = seg["start"]
        if start < start_min or start > start_max:
            continue
        end = min(start + target_length, total_duration)
        score = _window_density(segments, start, end)
        if score > best_score:
            best = (start, end)
            best_score = score

    if best_score < 0:
        for seg in segments:
            start = seg["start"]
            end = min(start + target_length, total_duration)
            score = _window_density(segments, start, end)
            if score > best_score:
                best = (start, end)
                best_score = score

    return best


def translate_blueprint_to_cut_plan(
    transcript: dict[str, Any],
    blueprint: dict[str, str],
) -> dict[str, Any]:
    segments = sorted(transcript.get("segments", []), key=lambda s: s["start"])
    if not segments:
        return {
            "start": 0.0,
            "end": 20.0,
            "focus": "auto",
            "subtitle_style": "normal",
        }

    total_duration = float(segments[-1]["end"])
    target_length = min(max(total_duration * 0.25, 15.0), 60.0)

    blueprint_text = " ".join(
        value for value in blueprint.values() if isinstance(value, str)
    ).lower()

    start_min = 0.0
    start_max = max(total_duration - target_length, 0.0)

    if any(token in blueprint_text for token in ["desfecho", "consequência", "resultado", "depois"]):
        start_min = total_duration * 0.6
        start_max = total_duration * 0.85
    elif any(token in blueprint_text for token in ["linha do tempo", "cronologia", "sequência", "sequencia"]):
        start_min = total_duration * 0.3
        start_max = total_duration * 0.6
    elif any(token in blueprint_text for token in ["origem", "contexto", "início", "inicio", "antes"]):
        start_min = 0.0
        start_max = total_duration * 0.4

    start, end = _best_window(segments, total_duration, target_length, start_min, start_max)

    if blueprint.get("ending") == "Cliffhanger":
        end_limit = total_duration * 0.9
        end = min(end, end_limit)
        if end - start < 15.0:
            start = max(end - 15.0, 0.0)

    if end - start < 15.0:
        end = min(start + 15.0, total_duration)

    if end - start > 60.0:
        end = start + 60.0

    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "focus": "auto",
        "subtitle_style": "normal",
    }


def generate_clip_sequence(
    transcript: dict[str, Any],
    blueprint: dict[str, str],
    platform: str,
) -> list[dict[str, Any]]:
    segments = sorted(transcript.get("segments", []), key=lambda s: s["start"])
    if not segments:
        return [translate_blueprint_to_cut_plan(transcript, blueprint) | {"role": "hook"}]

    total_duration = float(segments[-1]["end"])
    platform_max = {
        "tt": 40.0,
        "ig": 40.0,
        "yt": 45.0,
        "other": 40.0,
    }.get(platform, 40.0)

    if total_duration < 30:
        plan = translate_blueprint_to_cut_plan(transcript, blueprint)
        plan["role"] = "hook"
        return [plan]

    roles = ["hook", "context"]
    if total_duration >= 90:
        roles.append("payoff")
    if blueprint.get("ending") == "Cliffhanger" and total_duration >= 120:
        roles.append("cliffhanger")

    role_targets = {
        "hook": min(20.0, platform_max),
        "context": min(32.0, platform_max),
        "payoff": min(28.0, platform_max),
        "cliffhanger": min(22.0, platform_max),
    }

    windows: list[dict[str, Any]] = []

    def _overlaps(start: float, end: float) -> bool:
        for w in windows:
            overlap = min(end, w["end"]) - max(start, w["start"])
            if overlap > 0:
                duration = min(end - start, w["end"] - w["start"])
                if duration > 0 and overlap / duration > 0.35:
                    return True
        return False

    for role in roles:
        target = max(8.0, role_targets[role])
        start_min = 0.0
        start_max = max(total_duration - target, 0.0)

        if role == "hook":
            start_min = 0.0
            start_max = total_duration * 0.5
        elif role == "context":
            start_min = total_duration * 0.2
            start_max = total_duration * 0.6
        elif role == "payoff":
            start_min = total_duration * 0.5
            start_max = total_duration * 0.85
        elif role == "cliffhanger":
            start_min = total_duration * 0.6
            start_max = total_duration * 0.9

        start, end = _best_window(segments, total_duration, target, start_min, start_max)

        if end - start > platform_max:
            end = start + platform_max

        if _overlaps(start, end):
            continue

        windows.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "focus": "auto",
            "subtitle_style": "normal",
            "role": role,
        })

    if not windows:
        plan = translate_blueprint_to_cut_plan(transcript, blueprint)
        plan["role"] = "hook"
        windows.append(plan)

    return windows[:4]
