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
