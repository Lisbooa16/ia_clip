def expand_window(segs, idx, min_s, max_s, force_start=None):
    # anchor
    start = force_start if force_start is not None else segs[idx]["start"]
    end = segs[idx]["end"]

    # tenta puxar um segmento anterior (contexto) se couber
    if idx > 0:
        prev_start = segs[idx - 1]["start"]
        if end - prev_start <= max_s:
            start = prev_start

    # expande pra frente até max_s
    j = idx + 1
    while j < len(segs) and end - start < max_s:
        end_candidate = segs[j]["end"]
        if end_candidate - start > max_s:
            break
        end = end_candidate
        j += 1

    if end - start < min_s:
        return None, None

    return start, end


def generate_hook_caption(anchor_text: str, hooks: list = None):
    """
    Gera uma legenda viral baseada nos hooks detectados.
    """
    text = anchor_text.lower()
    hooks = hooks or []

    # Prioridade por tipo de hook detectado
    if any("fraud" in h for h in hooks):
        return "CUIDADO COM ESSE GOLPE! ⚠️"

    if any("money" in h for h in hooks):
        return "COMO FAZER DINHEIRO ASSIM 💸"

    if any("drama" in h for h in hooks):
        return "ISSO FOI EXPOSTO... 😱"

    if any("curiosity" in h for h in hooks):
        return "O QUE NINGUÉM TE CONTA..."

    if any("urgency" in h for h in hooks):
        return "ÚLTIMA CHANCE! ⏰"

    if any("social_proof" in h for h in hooks):
        return "TODO MUNDO TÁ VENDO ISSO 🔥"

    if any("clickbait" in h for h in hooks):
        return "VOCÊ NÃO VAI ACREDITAR 😳"

    # Fallback para palavras-chave no texto
    if any(w in text for w in ["quase", "merda", "morrer", "perigo"]):
        return "ISSO QUASE DEU MUITO ERRADO 😳"

    if any(w in text for w in ["ninguém", "nunca", "jamais"]):
        return "NINGUÉM FALA SOBRE ISSO…"

    if any(w in text for w in ["erro", "falha", "problema"]):
        return "ERA SÓ UM ERRO PRA ACABAR TUDO"

    return "OLHA ISSO 👀"