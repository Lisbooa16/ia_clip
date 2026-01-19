import re

WORDS_RE = re.compile(r"\w+")


def _lower(seg):
    return seg.get("text", "").lower()


def question(seg):
    text = seg.get("text", "").strip()
    return text.endswith("?")


def short_sentence(seg, max_words=10):
    return len(WORDS_RE.findall(seg.get("text", ""))) <= max_words


def pause(seg, max_duration=1.2):
    return (seg["end"] - seg["start"]) <= max_duration


def negation(seg):
    text = _lower(seg)
    return any(w in text for w in ["não", "nunca", "ninguém", "jamais", "no one", "never", "nobody"])


def numbers(seg):
    return any(ch.isdigit() for ch in seg.get("text", ""))


# ---- Novos hooks virais ----

def curiosity(seg):
    text = _lower(seg)
    patterns = [
        "ninguém te conta",
        "ninguém fala",
        "ninguém te disse",
        "você não sabia",
        "sabia disso",
        "segredo",
        "secret",
        "mistério",
        "misterio",
        "ninguém sabe",
        "o que ninguém te conta",
        "dormir na minha casa",   # <- pra esse tipo de situação
        "dormir na sua casa",
    ]
    return any(p in text for p in patterns)


def drama(seg):
    text = _lower(seg)
    patterns = [
        "traição", "traiu", "vazou", "vazamento", "exposed", "exposição",
        "cancelado", "cancelada", "polêmica", "escândalo", "escandalo",
        "destruiu", "acabou com", "acabou tudo", "briga", "treta",
    ]
    return any(p in text for p in patterns)


def money(seg):
    text = _lower(seg)
    patterns = [
        "dinheiro", "money", "ficar rico", "milionário", "milionaria",
        "ganhar grana", "ganhar dinheiro", "renda extra", "ficar rico",
        "lucro", "lucrar", "profit",
        "2 milhões", "2 milhoes", "dois milhões", "dois milhoes",
    ]
    return any(p in text for p in patterns)


def fraud(seg):
    text = _lower(seg)
    patterns = [
        "golpe", "fraude", "esquema", "pirâmide", "piramide",
        "scam", "fake", "enganou", "pegadinha", "furada",
    ]
    return any(p in text for p in patterns)


def urgency(seg):
    text = _lower(seg)
    patterns = [
        "agora", "right now", "última chance", "ultima chance",
        "antes que seja tarde", "antes que acabe",
        "corre", "hurry", "não perde", "não perca", "don't miss",
    ]
    return any(p in text for p in patterns)


def social_proof(seg):
    text = _lower(seg)
    patterns = [
        "todo mundo", "everyone", "milhões de", "milhoes de",
        "viral", "trending", "explodiu", "bombou",
        "famoso", "famosa", "celebridade", "influencer",
    ]
    return any(p in text for p in patterns)


def clickbait(seg):
    text = _lower(seg)
    patterns = [
        "você não vai acreditar", "you won't believe",
        "olha isso", "look at this",
        "isso mudou minha vida", "mudou minha vida",
        "isso vai explodir sua mente", "vai mudar tudo",
    ]
    return any(p in text for p in patterns)


HOOKS = {
    "question": question,
    "short_sentence": short_sentence,
    "pause": pause,
    "negation": negation,
    "numbers": numbers,
    "curiosity": curiosity,
    "drama": drama,
    "money": money,
    "fraud": fraud,
    "urgency": urgency,
    "social_proof": social_proof,
    "clickbait": clickbait,
}
