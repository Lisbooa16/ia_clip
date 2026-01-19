from .scoring import score_segment
from .expansion import expand_window, generate_hook_caption


def pick_viral_windows_generic(
    transcript,
    profile,
    min_s=35,
    max_s=45,
    top_k=6,
):
    segs = transcript.get("segments", [])
    candidates = []

    for i, seg in enumerate(segs):
        score, hooks = score_segment(seg, profile)

        # threshold mais suave, mas ainda filtrando ruído
        if score < 2.0:
            continue

        hook_start = segs[i]["start"]
        hook_end = segs[i]["end"]

        start, end = expand_window(
            segs,
            i,
            min_s,
            max_s,
            force_start=hook_start,
        )
        if start is None or end is None:
            continue
        duration = end - start

        if not (min_s <= duration <= max_s):
            continue

        # bônus por múltiplos hooks no mesmo segmento
        multi_hook_bonus = 0.3 * max(0, len(hooks) - 1)

        final_score = score + duration * 0.05 + multi_hook_bonus

        candidates.append({
            "start": start,
            "end": end,
            "anchor_start": hook_start,  # ← NOVO: onde o hook foi detectado
            "anchor_end": hook_end,      # ← NOVO: fim do segmento do hook
            "duration": duration,
            "score": final_score,
            "hooks": hooks,
            "anchor_text": seg["text"],
            "hook_caption": generate_hook_caption(seg["text"], hooks),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    MIN_GAP_BETWEEN_CLIPS = 60.0  # ← AUMENTADO: força clips bem separados

    final = []
    for c in candidates:
        # sem overlap
        overlaps = any(
            not (c["end"] <= f["start"] or c["start"] >= f["end"])
            for f in final
        )
        if overlaps:
            continue

        # evita clips colados; queremos cortes bem distintos do vídeo
        too_close = any(
            abs(c["start"] - f["start"]) < MIN_GAP_BETWEEN_CLIPS
            for f in final
        )
        if too_close:
            continue

        final.append(c)

        if len(final) >= top_k:
            break

    return sorted(final, key=lambda x: x["start"])