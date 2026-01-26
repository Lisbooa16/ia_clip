from __future__ import annotations

import json
import re
from pathlib import Path

from clips.models import VideoClip, ViralCandidate


CTA_LINE = "👉 Se inscreve no canal e comenta o que você achou 👇"

EMOTION_HOOKS = {
    "curiosity": "Você já reparou nisso?",
    "shock": "Isso chama atenção logo de cara.",
    "opinion": "Esse trecho divide opiniões.",
    "neutral": "Olha esse trecho.",
}


def generate_youtube_description(
    clip: VideoClip,
    viral_candidate: ViralCandidate | None = None,
) -> str:
    transcript_text = _load_transcript_text_for_clip(clip, viral_candidate)
    sentences = _split_sentences(transcript_text)
    hook = _pick_hook(sentences, viral_candidate)
    summary = _pick_summary(sentences, transcript_text)
    reason = _pick_reason_line(viral_candidate)

    lines = [hook, summary]
    if reason:
        lines.append(reason)
    lines.append(CTA_LINE)
    lines = [line for line in lines if line]

    if len(lines) < 3:
        lines.insert(1, "No trecho, o ponto principal aparece com clareza.")
    if len(lines) > 6:
        lines = lines[:6]

    return "\n".join(lines)


def generate_viral_caption(
    clip: VideoClip,
    viral_candidate: ViralCandidate | None = None,
) -> str:
    transcript_text = _load_transcript_text_for_clip(clip, viral_candidate)
    sentences = _split_sentences(transcript_text)
    hook = _pick_hook(sentences, viral_candidate)
    caption = hook

    if not caption:
        caption = "Tem um detalhe aqui que pouca gente comenta."

    if not caption.endswith("?"):
        caption = f"{caption} O que você acha?"

    if len(sentences) >= 2 and len(caption) < 140:
        follow = _trim_text(sentences[1], 120)
        if follow and follow not in caption:
            caption = f"{caption}\n{follow}"

    return caption.strip()


def _load_transcript_text_for_clip(
    clip: VideoClip,
    viral_candidate: ViralCandidate | None,
) -> str:
    if viral_candidate and viral_candidate.transcript_text:
        return viral_candidate.transcript_text

    job = clip.job
    transcript = job.transcript_data
    if not transcript or transcript == {"segments": "written_to_file"}:
        transcript = _load_transcript_from_path(job.transcript_path)

    if not isinstance(transcript, dict):
        return ""

    segments = transcript.get("segments", [])
    if not segments:
        return ""

    start = clip.start
    end = clip.end
    parts = []
    for seg in segments:
        try:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if seg_end <= start:
            continue
        if seg_start >= end:
            break
        text = (seg.get("text") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _load_transcript_from_path(path_value: str | None) -> dict | None:
    if not path_value:
        return None
    try:
        path = Path(path_value)
    except TypeError:
        return None
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _split_sentences(text: str) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def _pick_hook(sentences: list[str], viral_candidate: ViralCandidate | None) -> str:
    for sentence in sentences:
        if sentence.endswith("?"):
            return _trim_text(sentence, 140)

    emotion = "neutral"
    if viral_candidate and viral_candidate.emotion:
        emotion = viral_candidate.emotion

    prefix = EMOTION_HOOKS.get(emotion, EMOTION_HOOKS["neutral"])
    if sentences:
        return _trim_text(f"{prefix} {sentences[0]}", 140)
    return prefix


def _pick_summary(sentences: list[str], fallback: str) -> str:
    if len(sentences) >= 2:
        return _trim_text(f"Resumo rápido: {sentences[1]}", 180)
    if fallback:
        return _trim_text(f"Resumo rápido: {fallback}", 180)
    return "Resumo rápido: um recorte direto do trecho mais interessante."


def _pick_reason_line(viral_candidate: ViralCandidate | None) -> str | None:
    if not viral_candidate or not viral_candidate.reason:
        return None
    reason = viral_candidate.reason.strip()
    if not reason:
        return None
    return _trim_text(f"Por que chama atenção: {reason}", 180)


def _trim_text(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    shortened = cleaned[:limit]
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return shortened.strip() + "…"
