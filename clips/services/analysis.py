from __future__ import annotations

import json
import re
from pathlib import Path

from .keyword_signals import match_keywords


def load_transcript_for_job(job) -> dict | None:
    data = job.transcript_data
    if isinstance(data, dict) and data.get("segments") not in (None, "written_to_file"):
        return data

    transcript_path = job.transcript_path
    if not transcript_path:
        return None

    try:
        path = Path(transcript_path)
    except TypeError:
        return None

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def split_transcript_into_candidates(
    transcript: dict,
    min_duration: float = 15.0,
    max_duration: float = 40.0,
) -> list[dict]:
    segments = sorted(transcript.get("segments", []), key=lambda s: s.get("start", 0.0))
    if not segments:
        return []

    candidates = []
    current = []
    current_start = None
    current_end = None

    for idx, seg in enumerate(segments):
        try:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
        except (TypeError, ValueError):
            continue

        if seg_end <= seg_start:
            continue

        seg_duration = seg_end - seg_start
        if seg_duration >= max_duration:
            trimmed_end = seg_start + max_duration
            text = (seg.get("text") or "").strip()
            candidates.append(
                {
                    "start": seg_start,
                    "end": trimmed_end,
                    "duration": trimmed_end - seg_start,
                    "segments": [seg],
                    "text": text,
                }
            )
            current = []
            current_start = None
            current_end = None
            continue

        if current_start is None:
            current_start = seg_start
            current_end = seg_end
            current = [seg]
        else:
            current.append(seg)
            current_end = seg_end

        if current_start is None or current_end is None:
            continue

        duration = current_end - current_start
        next_seg = segments[idx + 1] if idx + 1 < len(segments) else None
        next_end = None
        if next_seg:
            try:
                next_end = float(next_seg.get("end", current_end))
            except (TypeError, ValueError):
                next_end = None

        should_close = duration >= min_duration and (
            next_end is None or (next_end - current_start) > max_duration
        )
        if should_close:
            text = " ".join((s.get("text") or "").strip() for s in current).strip()
            candidates.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "duration": duration,
                    "segments": list(current),
                    "text": text,
                }
            )
            current = []
            current_start = None
            current_end = None

    if current and current_start is not None and current_end is not None:
        duration = current_end - current_start
        if min_duration <= duration <= max_duration:
            text = " ".join((s.get("text") or "").strip() for s in current).strip()
            candidates.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "duration": duration,
                    "segments": list(current),
                    "text": text,
                }
            )

    return candidates


def score_candidate(text: str, duration: float) -> tuple[int, str, str]:
    score = 0
    reasons = []

    matches = match_keywords(
        text,
        categories=["entity", "emotion", "opinion", "curiosity", "shock"],
    )

    entity_hits = _extract_entities(text)
    if entity_hits:
        score += 20
        reasons.append("Menciona nomes ou entidades relevantes.")

    if matches.get("emotion"):
        score += 15
        reasons.append("Contém termos emocionais fortes.")

    if _contains_question(text):
        score += 10
        reasons.append("Usa pergunta direta.")

    if matches.get("opinion"):
        score += 20
        reasons.append("Mostra opinião, conflito ou controvérsia.")

    if 20.0 <= duration <= 35.0:
        score += 15
        reasons.append("Duração ideal entre 20 e 35s.")

    words_per_second = _words_per_second(text, duration)
    if words_per_second >= 2.8:
        score += 10
        reasons.append(f"Fala densa ({words_per_second:.1f} palavras/s).")

    score = min(score, 100)
    emotion = _classify_emotion(text)
    reason = " ".join(reasons) if reasons else "Segmento neutro, sem sinais fortes."
    return score, emotion, reason


def _extract_entities(text: str) -> list[str]:
    if not text:
        return []
    hits = []
    for token in re.findall(r"\b[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ]{2,}\b", text):
        hits.append(token)
    matches = match_keywords(text, categories=["entity"])
    for hint in matches.get("entity", []):
        hits.append(hint)
    return list(dict.fromkeys(hits))


def _contains_question(text: str) -> bool:
    lowered = (text or "").lower()
    if "?" in lowered:
        return True
    return False


def _words_per_second(text: str, duration: float) -> float:
    if duration <= 0:
        return 0.0
    words = re.findall(r"\w+", text or "")
    return len(words) / duration


def _classify_emotion(text: str) -> str:
    matches = match_keywords(text, categories=["shock", "curiosity", "opinion"])
    if matches.get("shock"):
        return "shock"
    if _contains_question(text) or matches.get("curiosity"):
        return "curiosity"
    if matches.get("opinion"):
        return "opinion"
    return "neutral"
