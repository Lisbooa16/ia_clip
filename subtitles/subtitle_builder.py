from __future__ import annotations

import math
import re
from pathlib import Path

import pysubs2

from subtitles.caption_styles import (
    CaptionStyle,
    CaptionStyleConfig,
    build_force_style,
    normalize_config,
    normalize_style,
    alignment_for_position,
    margins_for_position,
    to_ass_color,
)


def clamp_segment(seg, clip_start, clip_end):
    return {
        "start": max(seg["start"], clip_start) - clip_start,
        "end": min(seg["end"], clip_end) - clip_start,
        "text": seg["text"].strip() or "…"
    }


def segments_for_clip(all_segments, clip_start, clip_end):
    out = []

    for seg in all_segments:
        if seg["end"] <= clip_start:
            continue
        if seg["start"] >= clip_end:
            break

        out.append(clamp_segment(seg, clip_start, clip_end))

    return out


def fill_gaps(segments, max_gap=0.6):
    filled = []
    last_end = 0.0

    for seg in segments:
        if seg["start"] - last_end > max_gap:
            filled.append({
                "start": last_end,
                "end": seg["start"],
                "text": "…"
            })
        filled.append(seg)
        last_end = seg["end"]

    return filled


def words_for_clip(transcript, clip_start, clip_end, max_gap=0.5):
    words = []
    segments = transcript.get("segments", []) if isinstance(transcript, dict) else transcript

    for s in segments:
        for w in s.get("words", []):
            if w["end"] <= clip_start:
                continue
            if w["start"] >= clip_end:
                continue

            words.append({
                "start": max(0.0, w["start"] - clip_start),
                "end": min(clip_end, w["end"]) - clip_start,
                "word": w["word"] or "…",
            })

    if not words:
        return [{
            "start": 0.0,
            "end": clip_end - clip_start,
            "word": "…",
        }]

    words.sort(key=lambda w: w["start"])

    filled = []
    last_end = 0.0

    for w in words:
        if w["start"] - last_end > max_gap:
            filled.append({
                "start": last_end,
                "end": w["start"],
                "word": "…",
            })
        filled.append(w)
        last_end = w["end"]

    total_dur = clip_end - clip_start
    if total_dur - last_end > max_gap:
        filled.append({
            "start": last_end,
            "end": total_dur,
            "word": "…",
        })

    return filled


def _split_caption_text(text, max_chars=14, min_chars=6):
    words = text.split()
    if len(words) <= 1:
        return text, [len(text)]

    candidates = []
    for idx in range(1, len(words)):
        line1 = " ".join(words[:idx])
        line2 = " ".join(words[idx:])
        len1 = len(line1)
        len2 = len(line2)
        if len1 <= max_chars and len2 <= max_chars and len2 >= min_chars:
            penalty = abs(len1 - len2) + (0.5 if len2 > len1 else 0.0)
            candidates.append((penalty, line1, line2, len1, len2))

    if candidates:
        _, line1, line2, len1, len2 = sorted(candidates, key=lambda c: c[0])[0]
        return f"{line1}\\N{line2}", [len1, len2]

    best = None
    for idx in range(1, len(words)):
        line1 = " ".join(words[:idx])
        line2 = " ".join(words[idx:])
        len1 = len(line1)
        len2 = len(line2)
        score = max(len1, len2)
        if best is None or score < best[0]:
            best = (score, line1, line2, len1, len2)

    if best:
        _, line1, line2, len1, len2 = best
        return f"{line1}\\N{line2}", [len1, len2]

    return text, [len(text)]


def _apply_font_override(text, target_size):
    return f"{{\\fs{target_size}}}{text}"


def to_srt(segments):
    def ts(t):
        ms = int(t * 1000)
        h = ms // 3600000
        m = (ms % 3600000) // 60000
        s = (ms % 60000) // 1000
        ms %= 1000
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    out = []
    for i, s in enumerate(segments, 1):
        original_text = s["text"]
        if original_text != "…":
            base_max = 14
            broken_text, line_lengths = _split_caption_text(original_text, max_chars=base_max, min_chars=6)
            max_len = max(line_lengths) if line_lengths else len(original_text)
            font_size = 44
            if max_len > base_max:
                for candidate in (42, 40, 38, 36):
                    font_size = candidate
                    allowed = int(base_max * 44 / candidate)
                    if max_len <= allowed:
                        break
                broken_text = _apply_font_override(broken_text, font_size)
            if broken_text != original_text:
                print(
                    "[SUB] 🧩 "
                    f"orig='{original_text}' "
                    f"final='{broken_text}' "
                    f"size={font_size} "
                    f"lens={line_lengths}"
                )
            s = dict(s)
            s["text"] = broken_text
        out.append(str(i))
        out.append(f"{ts(s['start'])} --> {ts(s['end'])}")
        out.append(s["text"])
        out.append("")

    return "\n".join(out)


def _split_words_for_lines(words, max_chars=22):
    if len(words) <= 3:
        return [words]
    total_chars = sum(len(w["word"]) for w in words) + max(len(words) - 1, 0)
    if total_chars <= max_chars:
        return [words]
    target = total_chars / 2
    running = 0
    split_idx = 1
    for idx, w in enumerate(words[:-1], 1):
        running += len(w["word"]) + 1
        if running >= target:
            split_idx = idx
            break
    return [words[:split_idx], words[split_idx:]]


def build_word_by_word_ass(
    words: list[dict],
    config: CaptionStyleConfig,
    out_ass_path: str,
):
    subs = pysubs2.SSAFile()
    style = pysubs2.SSAStyle()
    style.fontname = config.font_family
    style.fontsize = max(24, int(config.font_size))
    style.primarycolor = pysubs2.Color.from_ass(to_ass_color(config.highlight_color))
    style.secondarycolor = pysubs2.Color.from_ass(to_ass_color(config.font_color))
    style.outlinecolor = pysubs2.Color.from_ass("&H00000000")
    style.backcolor = pysubs2.Color.from_ass(
        to_ass_color("#000000", "80" if config.background else "00")
    )
    style.bold = True
    style.outline = 2
    style.shadow = 1
    style.alignment = alignment_for_position(config.position)
    margin_l, margin_r, margin_v = margins_for_position(config.position)
    style.marginl = margin_l
    style.marginr = margin_r
    style.marginv = margin_v
    style.borderstyle = 3 if config.background else 1
    subs.styles["Karaoke"] = style

    phrases = []
    buf = []
    start = None
    end = None
    char_count = 0

    for w in words:
        word = (w.get("word") or "").strip()
        if not word:
            continue
        if start is None:
            start = w["start"]
        end = w["end"]
        char_count += len(word) + (1 if buf else 0)
        buf.append(w)
        duration = end - start
        if duration >= 2.4 or char_count >= 28 or re.search(r"[.!?]$", word):
            phrases.append((buf, start, end))
            buf = []
            start = None
            end = None
            char_count = 0

    if buf and start is not None and end is not None:
        phrases.append((buf, start, end))

    for phrase_words, phrase_start, phrase_end in phrases:
        lines = _split_words_for_lines(phrase_words)
        line_texts = []
        for line in lines:
            tokens = []
            for w in line:
                start_t = float(w["start"])
                end_t = float(w["end"])
                dur = max(end_t - start_t, 0.05)
                dur_cs = max(int(round(dur * 100)), 1)
                tokens.append(f"{{\\k{dur_cs}}}{w['word']}")
            line_texts.append(" ".join(tokens))
        text = "\\N".join(line_texts)
        event = pysubs2.SSAEvent(
            start=int(math.floor(phrase_start * 1000)),
            end=int(math.ceil(phrase_end * 1000)),
            text=text,
            style="Karaoke",
        )
        subs.events.append(event)

    subs.save(out_ass_path)


def build_subtitle_artifacts(
    transcript: dict,
    clip_start: float,
    clip_end: float,
    caption_style: str | CaptionStyle | None,
    caption_config: dict | None,
    output_dir: Path,
    clip_id: str,
    suffix: str = "",
) -> tuple[Path, CaptionStyle, CaptionStyleConfig]:
    style = normalize_style(caption_style)
    config = normalize_config(caption_config)

    has_word_timestamps = any(
        bool(seg.get("words"))
        for seg in transcript.get("segments", [])
    )
    if style == CaptionStyle.WORD_BY_WORD and not has_word_timestamps:
        style = CaptionStyle.STATIC

    output_dir.mkdir(parents=True, exist_ok=True)
    if style == CaptionStyle.WORD_BY_WORD:
        words = words_for_clip(transcript, clip_start, clip_end)
        out_path = output_dir / f"{clip_id}{suffix}.ass"
        build_word_by_word_ass(words, config, str(out_path))
    else:
        segs = segments_for_clip(transcript["segments"], clip_start, clip_end)
        segs = fill_gaps(segs)
        srt_text = to_srt(segs)
        out_path = output_dir / f"{clip_id}{suffix}.srt"
        out_path.write_text(srt_text, encoding="utf-8")

    return out_path, style, config


def build_subtitle_filter(
    subtitle_path: str,
    caption_style: str | CaptionStyle | None,
    caption_config: dict | None,
    play_res: tuple[int, int] = (1080, 1920),
) -> str | None:
    if not subtitle_path:
        return None
    subtitle_file = Path(subtitle_path)
    if not subtitle_file.exists():
        print(f"[SUB] ⚠️ missing subtitles path={subtitle_path}")
        return None
    sub_path = subtitle_path.replace("\\", "/").replace(":", "\\:")
    style = normalize_style(caption_style)
    config = normalize_config(caption_config)
    if style == CaptionStyle.STATIC:
        force_style = build_force_style(config, play_res)
        return f"subtitles=filename='{sub_path}':force_style='{force_style}'"
    return f"subtitles=filename='{sub_path}'"
