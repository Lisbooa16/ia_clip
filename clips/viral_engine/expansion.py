def expand_window(segments, anchor_idx, min_s=8, max_s=20):
    """
    Expande a janela ao redor da âncora (momento viral).
    Retorna um dict com start, end, anchor_start, anchor_end, score, hooks, caption.
    """
    if not segments or anchor_idx >= len(segments):
        return None

    anchor = segments[anchor_idx]

    # Começar pela âncora
    left = anchor_idx
    right = anchor_idx

    # Expandir para trás (contexto) - até 2-3 segundos antes
    while left > 0:
        prev_seg = segments[left - 1]
        if anchor["start"] - prev_seg["start"] > 3.0:  # Máximo 3s de contexto
            break
        left -= 1

    # Expandir para frente até atingir min_s
    while right < len(segments) - 1:
        window_dur = segments[right]["end"] - segments[left]["start"]
        if window_dur >= min_s:
            break
        right += 1

    # Tentar expandir mais até max_s se melhorar o score
    best_right = right
    best_score = sum(s["score"] for s in segments[left:right + 1])

    while right < len(segments) - 1:
        window_dur = segments[right + 1]["end"] - segments[left]["start"]
        if window_dur > max_s:
            break

        right += 1
        current_score = sum(s["score"] for s in segments[left:right + 1])
        if current_score > best_score:
            best_score = current_score
            best_right = right

    right = best_right

    # Construir a janela final
    window_segments = segments[left:right + 1]

    # Coletar todos os hooks detectados
    all_hooks = []
    for seg in window_segments:
        all_hooks.extend(seg.get("hooks", []))

    return {
        "start": window_segments[0]["start"],
        "end": window_segments[-1]["end"],
        "anchor_start": anchor["start"],  # ← IMPORTANTE
        "anchor_end": anchor["end"],  # ← IMPORTANTE
        "score": best_score,
        "hooks": list(set(all_hooks)),  # Remove duplicatas
        "caption": generate_hook_caption(window_segments, all_hooks),
    }


def generate_hook_caption(segments, hooks):
    """
    Gera uma caption viral baseada nos hooks detectados.
    Prioriza o texto da âncora (primeiro segmento com hook).
    """
    if not segments:
        return ""

    # Se temos hooks, usar o texto do segmento que contém o hook mais forte
    if hooks:
        for seg in segments:
            seg_hooks = seg.get("hooks", [])
            if any(h in seg_hooks for h in hooks):
                return seg["text"][:120].strip()

    # Fallback: usar o primeiro segmento
    return segments[0]["text"][:120].strip()
