from __future__ import annotations

import re


def build_editorial_overlays(
    transcript_text: str,
    clip_start: float,
    clip_end: float,
    title: str | None = None,
) -> list[dict]:
    sentences = _split_sentences(transcript_text)
    if not sentences:
        return []

    hook = _pick_sentence(sentences, prefer_question=True)
    context = _pick_sentence(sentences[1:] if len(sentences) > 1 else sentences)
    closing = sentences[-1]

    if title:
        hook = f"{title.strip()} — {hook}"

    hook = _trim_text(hook)
    context = _trim_text(context)
    closing = _trim_text(closing)

    duration = max(clip_end - clip_start, 0.0)
    if duration <= 0:
        return []

    hook_dur = _clamp(duration * 0.18, 1.5, 3.0)
    context_dur = _clamp(duration * 0.2, 1.5, 3.5)
    closing_dur = _clamp(duration * 0.18, 1.5, 3.0)

    hook_start = clip_start
    hook_end = min(clip_start + hook_dur, clip_end)
    context_start = min(hook_end + 0.1, clip_end)
    context_end = min(context_start + context_dur, clip_end)
    closing_end = clip_end
    closing_start = max(clip_end - closing_dur, context_end + 0.1)

    overlays = []
    if hook and hook_end - hook_start >= 0.6:
        overlays.append(_build_segment(hook_start, hook_end, hook))
    if context and context_end - context_start >= 0.6 and context_start < closing_start:
        overlays.append(_build_segment(context_start, context_end, context))
    if closing and closing_end - closing_start >= 0.6:
        overlays.append(_build_segment(closing_start, closing_end, closing))

    return overlays


def merge_transcript_overlays(transcript: dict, overlays: list[dict]) -> dict:
    if not overlays:
        return transcript
    merged = list(transcript.get("segments", []))
    merged.extend(overlays)
    merged.sort(key=lambda s: s.get("start", 0.0))
    return {"segments": merged}


def _build_segment(start: float, end: float, text: str) -> dict:
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "text": text.strip(),
        "words": [],
    }


def _split_sentences(text: str) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def _pick_sentence(sentences: list[str], prefer_question: bool = False) -> str:
    if prefer_question:
        for sentence in sentences:
            if sentence.endswith("?"):
                return sentence
    return sentences[0]


def _trim_text(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    shortened = text[:limit]
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return shortened.strip() + "…"


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))
