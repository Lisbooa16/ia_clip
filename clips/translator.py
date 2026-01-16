from __future__ import annotations

from typing import Any


_SECTION_ORDER = ["opening", "setup", "context", "tension", "reveal", "ending"]
_FILLER_TOKENS = {
    "uh",
    "um",
    "erm",
    "tipo",
    "tipo assim",
    "tipo",
    "então",
    "entao",
    "bom",
    "né",
    "ne",
    "assim",
}


def _tokenize(text: str) -> list[str]:
    cleaned = []
    for raw in (text or "").lower().replace("\n", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if token and token not in _FILLER_TOKENS:
            cleaned.append(token)
    return cleaned


def _segment_score(segment: dict[str, Any], keywords: set[str]) -> float:
    text = (segment.get("text") or "").lower()
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    overlap = len(set(tokens) & keywords)
    density = _segment_word_count(segment) / max(segment["end"] - segment["start"], 0.3)
    filler_penalty = 0.5 if any(token in _FILLER_TOKENS for token in tokens[:3]) else 1.0
    return (overlap * 2.0 + density) * filler_penalty


def _find_best_segment(
    segments: list[dict[str, Any]],
    start_time: float,
    end_time: float,
    keywords: set[str],
) -> dict[str, Any] | None:
    best_seg = None
    best_score = -1.0
    for seg in segments:
        if seg["end"] <= start_time:
            continue
        if seg["start"] >= end_time:
            break
        score = _segment_score(seg, keywords)
        if score > best_score:
            best_score = score
            best_seg = seg
    return best_seg


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
    target_length = min(max(total_duration * 0.22, 15.0), 60.0)
    start_min = 0.0
    start_max = max(total_duration - target_length, 0.0)

    opening_keywords = set(_tokenize(blueprint.get("opening", "")))
    if not opening_keywords:
        opening_keywords = set(_tokenize(" ".join(
            value for key, value in blueprint.items() if key != "ending" and isinstance(value, str)
        )))

    opening_seg = _find_best_segment(
        segments,
        start_min,
        min(total_duration * 0.5, start_max + target_length),
        opening_keywords,
    )

    if opening_seg:
        start = opening_seg["start"]
        end = min(start + target_length, total_duration)
    else:
        start, end = _best_window(segments, total_duration, target_length, start_min, start_max)

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

    section_texts = {
        "opening": blueprint.get("opening", ""),
        "setup": blueprint.get("setup", ""),
        "context": blueprint.get("context", ""),
        "tension": blueprint.get("tension", ""),
        "reveal": blueprint.get("reveal", ""),
        "ending": blueprint.get("ending", ""),
    }

    section_count = len(_SECTION_ORDER)
    target_per_section = max(min(total_duration / section_count, platform_max), 6.0)
    windows: list[dict[str, Any]] = []
    cursor = 0.0

    for idx, section in enumerate(_SECTION_ORDER):
        target = min(target_per_section, platform_max)
        remaining = max(total_duration - cursor, 0.0)
        if idx == section_count - 1:
            target = min(remaining, platform_max)

        keywords = set(_tokenize(section_texts.get(section, "")))
        search_start = cursor
        search_end = min(total_duration, cursor + max(platform_max, target * 2))
        best_seg = _find_best_segment(segments, search_start, search_end, keywords)

        if best_seg is None:
            start = cursor
        else:
            start = max(cursor, best_seg["start"])

        end = min(start + target, total_duration)
        if end <= start:
            end = min(start + 6.0, total_duration)

        print(f"[BLUEPRINT] {section} -> {start:.2f}-{end:.2f}")

        windows.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "focus": "auto",
            "subtitle_style": "normal",
            "role": section,
        })
        cursor = end

    if not windows:
        plan = translate_blueprint_to_cut_plan(transcript, blueprint)
        plan["role"] = "opening"
        windows.append(plan)

    return windows
