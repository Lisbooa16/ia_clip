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
