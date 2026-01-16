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
        out.append(str(i))
        out.append(f"{ts(s['start'])} --> {ts(s['end'])}")
        out.append(s["text"])
        out.append("")

    return "\n".join(out)
