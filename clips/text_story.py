from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class StoryPart:
    index: int
    text: str
    estimated_duration: float


SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
CLAUSE_RE = re.compile(r"(?<=[,;:])\s+")


def split_into_sentences(text: str) -> list[str]:
    clean = " ".join(text.strip().split())
    if not clean:
        return []
    sentences = [s.strip() for s in SENTENCE_RE.findall(clean) if s.strip()]
    return sentences or [clean]


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _estimate_duration(words: int, sentences: int, words_per_sec: float, pause_s: float) -> float:
    if words == 0:
        return 0.0
    return (words / words_per_sec) + max(0, sentences - 1) * pause_s


def _split_long_sentence(sentence: str, max_words: int) -> list[str]:
    if max_words <= 0:
        return [sentence]
    words = [w for w in sentence.split() if w.strip()]
    if len(words) <= max_words:
        return [sentence]
    parts = [p.strip() for p in CLAUSE_RE.split(sentence) if p.strip()]
    if len(parts) > 1:
        return parts
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks or [sentence]


def split_story_text(
    text: str,
    max_duration_s: float = 60.0,
    words_per_sec: float = 2.5,
    pause_s: float = 0.0,
) -> list[StoryPart]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    max_duration_s = min(60.0, float(max_duration_s))
    max_words = int(max_duration_s * words_per_sec)
    expanded = []
    for sentence in sentences:
        expanded.extend(_split_long_sentence(sentence, max_words))
    sentences = expanded

    parts: list[str] = []
    current: list[str] = []
    current_words = 0
    current_sentences = 0
    target_duration_s = 45.0
    min_duration_s = 20.0

    def flush_current():
        nonlocal current, current_words, current_sentences
        if current:
            parts.append(" ".join(current).strip())
        current = []
        current_words = 0
        current_sentences = 0

    for sentence in sentences:
        words = [w for w in sentence.split() if w.strip()]
        if not words:
            continue
        sentence_words = len(words)
        if current:
            next_words = current_words + sentence_words
            next_sentences = current_sentences + 1
            next_duration = _estimate_duration(
                next_words,
                next_sentences,
                words_per_sec,
                pause_s,
            )
            if next_duration > max_duration_s:
                flush_current()
        current.append(sentence)
        current_words += sentence_words
        current_sentences += 1
        current_duration = _estimate_duration(
            current_words,
            current_sentences,
            words_per_sec,
            pause_s,
        )
        if current_duration >= target_duration_s:
            flush_current()

    flush_current()

    idx = 0
    while idx < len(parts) - 1:
        duration = _estimate_duration(
            _word_count(parts[idx]),
            len(split_into_sentences(parts[idx])),
            words_per_sec,
            pause_s,
        )
        if duration < min_duration_s:
            merged = f"{parts[idx]} {parts[idx + 1]}".strip()
            merged_duration = _estimate_duration(
                _word_count(merged),
                len(split_into_sentences(merged)),
                words_per_sec,
                pause_s,
            )
            if merged_duration <= max_duration_s:
                parts[idx] = merged
                parts.pop(idx + 1)
                continue
            if idx > 0:
                merged_prev = f"{parts[idx - 1]} {parts[idx]}".strip()
                merged_prev_duration = _estimate_duration(
                    _word_count(merged_prev),
                    len(split_into_sentences(merged_prev)),
                    words_per_sec,
                    pause_s,
                )
                if merged_prev_duration <= max_duration_s:
                    parts[idx - 1] = merged_prev
                    parts.pop(idx)
                    idx = max(idx - 1, 0)
                    continue
        idx += 1

    story_parts: list[StoryPart] = []
    for idx, part in enumerate(parts, start=1):
        sentences_count = len(split_into_sentences(part))
        word_count = _word_count(part)
        duration = _estimate_duration(word_count, sentences_count, words_per_sec, pause_s)
        story_parts.append(
            StoryPart(
                index=idx,
                text=part,
                estimated_duration=duration,
            )
        )

    return story_parts


def build_transcript_from_text(
    text: str,
    words_per_sec: float = 2.5,
    pause_s: float = 0.0,
) -> dict:
    sentences = split_into_sentences(text)
    segments: list[dict] = []
    current_time = 0.0

    for idx, sentence in enumerate(sentences):
        words = [w for w in sentence.split() if w.strip()]
        if not words:
            continue
        word_duration = 1.0 / words_per_sec
        segment_start = current_time
        words_payload = []
        for word in words:
            word_start = current_time
            current_time = round(current_time + word_duration, 3)
            words_payload.append(
                {
                    "start": word_start,
                    "end": current_time,
                    "word": word,
                }
            )
        segment_end = current_time
        segments.append(
            {
                "start": segment_start,
                "end": segment_end,
                "text": sentence.strip(),
                "words": words_payload,
            }
        )
        if idx < len(sentences) - 1:
            current_time = round(current_time + pause_s, 3)

    return {"segments": segments}
