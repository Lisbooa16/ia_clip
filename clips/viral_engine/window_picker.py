from .scoring import score_segment
from .expansion import expand_window, generate_hook_caption


def pick_viral_windows_generic(
    transcript,
    profile,
    min_s=18,
    max_s=30,
    top_k=6,
):
    segs = transcript.get("segments", [])
    candidates = []

    for i, seg in enumerate(segs):
        score, hooks = score_segment(seg, profile)
        if score < 2.5:
            continue

        hook_start = segs[i]["start"]

        start, end = expand_window(
            segs,
            i,
            min_s,
            max_s,
            force_start=hook_start
        )
        if start is None or end is None:
            continue
        duration = end - start

        if not (min_s <= duration <= max_s):
            continue

        candidates.append({
            "start": start,
            "end": end,
            "duration": duration,
            "score": score + duration * 0.05,
            "hooks": hooks,
            "anchor_text": seg["text"],
            "hook_caption": generate_hook_caption(seg["text"]),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    final = []
    for c in candidates:
        if all(c["end"] <= f["start"] or c["start"] >= f["end"] for f in final):
            final.append(c)
        if len(final) >= top_k:
            break

    return sorted(final, key=lambda x: x["start"])
