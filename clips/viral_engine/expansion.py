def expand_window(segs, idx, min_s, max_s, force_start=None):
    start = force_start if force_start is not None else segs[idx]["start"]
    end = start

    j = idx
    while j < len(segs) and end - start < max_s:
        end = segs[j]["end"]
        j += 1

    if end - start < min_s:
        return None, None

    return start, end


def generate_hook_caption(anchor_text: str):
    text = anchor_text.lower()

    if any(w in text for w in ["quase", "merda", "morrer", "perigo"]):
        return "ISSO QUASE DEU MUITO ERRADO 😳"

    if any(w in text for w in ["ninguém", "nunca", "jamais"]):
        return "NINGUÉM FALA SOBRE ISSO…"

    if any(w in text for w in ["erro", "falha", "problema"]):
        return "ERA SÓ UM ERRO PRA ACABAR TUDO"

    return "OLHA ISSO 👀"