from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class StoryPart:
    index: int
    text: str
    estimated_duration: float


SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")


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


def _chunk_words(words: list[str], max_words: int) -> list[list[str]]:
    if max_words <= 0:
        return [words]
    return [words[i:i + max_words] for i in range(0, len(words), max_words)]


def split_story_text(
    text: str,
    max_duration_s: float = 55.0,
    words_per_sec: float = 2.6,
    pause_s: float = 0.15,
) -> list[StoryPart]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    parts: list[str] = []
    current: list[str] = []
    current_words = 0
    current_sentences = 0
    max_words = int(max_duration_s * words_per_sec)

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
        if len(words) > max_words:
            flush_current()
            for chunk in _chunk_words(words, max_words):
                parts.append(" ".join(chunk).strip())
            continue

        next_words = current_words + len(words)
        next_sentences = current_sentences + 1
        if current and _estimate_duration(next_words, next_sentences, words_per_sec, pause_s) > max_duration_s:
            flush_current()
        current.append(sentence)
        current_words += len(words)
        current_sentences += 1

    flush_current()

    story_parts: list[StoryPart] = []
    for idx, part in enumerate(parts, start=1):
        part_text = part
        if idx < len(parts):
            part_text = f"{part_text} Continua no PT{idx + 1}."
        sentences_count = len(split_into_sentences(part_text))
        word_count = _word_count(part_text)
        duration = _estimate_duration(word_count, sentences_count, words_per_sec, pause_s)
        story_parts.append(
            StoryPart(
                index=idx,
                text=part_text,
                estimated_duration=duration,
            )
        )

    return story_parts


def build_transcript_from_text(
    text: str,
    words_per_sec: float = 2.6,
    pause_s: float = 0.15,
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
