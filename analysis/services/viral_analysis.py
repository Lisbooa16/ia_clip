from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import re
import yt_dlp

from analysis.utils import calculate_engagement, score_duration, score_engagement


PLATFORM_HOSTS = {
    "youtube": ("youtube.com", "youtu.be"),
    "tiktok": ("tiktok.com",),
    "instagram": ("instagram.com",),
}

STOPWORDS = {
    "a", "o", "os", "as", "de", "da", "do", "das", "dos", "em", "no", "na",
    "nos", "nas", "e", "ou", "para", "por", "com", "sem", "um", "uma", "uns",
    "umas", "que", "como", "quando", "onde", "porque", "porquê", "sobre", "até",
    "entre", "após", "antes", "mais", "menos", "já", "não", "sim", "ao", "aos",
    "às", "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "without", "from", "by", "is", "are", "was", "were", "be", "been", "this",
    "that", "these", "those", "you", "your", "we", "our", "they", "their",
}

ARCHETYPE_RULES = [
    {
        "keywords": {"desapareceu", "sumiu", "mistério", "misterio", "desaparecimento"},
        "archetype": "Mistério / Desaparecimento",
        "category": "história real / mistério",
        "motivations": ["curiosidade", "tensão"],
        "moral": "ausência de resposta clara e risco pessoal",
    },
    {
        "keywords": {"golpe", "fraude", "esquema", "pirâmide", "piramide"},
        "archetype": "Engano / Fraude",
        "category": "escândalo / denúncia",
        "motivations": ["choque", "alerta"],
        "moral": "quebra de confiança e impacto social",
    },
    {
        "keywords": {"traição", "traiu", "enganou", "vazou"},
        "archetype": "Traição / Ruptura",
        "category": "conflito pessoal",
        "motivations": ["curiosidade", "drama"],
        "moral": "ruptura de alianças e consequência íntima",
    },
    {
        "keywords": {"prisão", "preso", "condenado", "tribunal", "crime"},
        "archetype": "Queda / Consequência",
        "category": "crime-adjacent",
        "motivations": ["curiosidade", "justiça"],
        "moral": "consequência pública e responsabilidade",
    },
    {
        "keywords": {"fuga", "escapou", "resgate", "capturado"},
        "archetype": "Escape / Sobrevivência",
        "category": "ação / sobrevivência",
        "motivations": ["tensão", "adrenalina"],
        "moral": "instinto de sobrevivência em risco real",
    },
    {
        "keywords": {"demissão", "demitido", "cancelado", "polêmica", "polêmica"},
        "archetype": "Queda Pública",
        "category": "controverso",
        "motivations": ["curiosidade", "conflito"],
        "moral": "efeito dominó da opinião pública",
    },
]

EMOTION_KEYWORDS = {
    "curiosidade": {"mistério", "segredo", "revelado", "descoberta"},
    "surpresa": {"surpresa", "chocante", "incrível", "inesperado", "absurdo"},
    "tensão": {"tensão", "risco", "perigo", "ameaça", "urgente"},
    "empatia": {"emocionante", "comovente", "história", "superação", "família"},
    "indignação": {"polêmica", "injustiça", "revolta", "traição", "denúncia"},
    "medo": {"ameaça", "risco", "assustador", "perigo"},
}


@dataclass
class ViralAnalysisResult:
    platform: str
    metadata: dict[str, Any]
    score: int
    score_label: str
    confidence_score: float
    confidence_label: str
    narrative_archetype: str
    viral_category: str
    audience_motivation: list[str]
    emotional_triggers: list[str]
    story_insights: dict[str, str]
    similar_videos: list[dict[str, Any]]
    evidence_videos: list[dict[str, Any]]
    editorial_decisions: list[dict[str, Any]]
    sequence_preview: list[dict[str, Any]]
    discarded_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "metadata": self.metadata,
            "score": self.score,
            "score_label": self.score_label,
            "confidence_score": self.confidence_score,
            "confidence_label": self.confidence_label,
            "narrative_archetype": self.narrative_archetype,
            "viral_category": self.viral_category,
            "audience_motivation": self.audience_motivation,
            "emotional_triggers": self.emotional_triggers,
            "story_insights": self.story_insights,
            "similar_videos": self.similar_videos,
            "evidence_videos": self.evidence_videos,
            "editorial_decisions": self.editorial_decisions,
            "sequence_preview": self.sequence_preview,
            "discarded_reason": self.discarded_reason,
        }


def detect_platform(url: str) -> str:
    host = urlparse(url).netloc.lower()
    for platform, hosts in PLATFORM_HOSTS.items():
        if any(h in host for h in hosts):
            return platform
    return "desconhecida"


def fetch_metadata(url: str) -> dict[str, Any]:
    options = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "extract_flat": False,
        "nocheckcertificate": True,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)

    if not info:
        return {}

    return {
        "title": info.get("title"),
        "description": info.get("description"),
        "duration": info.get("duration"),
        "views": info.get("view_count"),
        "likes": info.get("like_count"),
        "comments": info.get("comment_count"),
        "uploader": info.get("uploader"),
        "tags": info.get("tags") or [],
        "upload_date": info.get("upload_date"),
        "thumbnail": info.get("thumbnail"),
    }


def search_similar_videos(query: str, limit: int = 6) -> list[dict[str, Any]]:
    if not query:
        return []

    options = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "dump_single_json": True,
        "nocheckcertificate": True,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        data = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)

    results = []
    for entry in data.get("entries", []) if data else []:
        if not entry:
            continue
        results.append(
            {
                "title": entry.get("title") or "",
                "url": entry.get("url") or entry.get("webpage_url") or "",
                "views": entry.get("view_count"),
                "thumbnail": entry.get("thumbnail"),
                "uploader": entry.get("uploader"),
                "duration": entry.get("duration"),
                "platform": "youtube",
            }
        )

    return results


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\sáéíóúàâêôãõçÁÉÍÓÚÀÂÊÔÃÕÇ]", " ", text)
    return [token for token in cleaned.split() if token]


def _extract_keywords(text: str, limit: int = 8) -> list[str]:
    tokens = [t.lower() for t in _tokenize(text)]
    freq: dict[str, int] = {}
    for token in tokens:
        if len(token) < 4 or token in STOPWORDS:
            continue
        freq[token] = freq.get(token, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda item: item[1], reverse=True)
    return [token for token, _count in sorted_tokens[:limit]]


def _extract_entities(title: str) -> list[str]:
    entities = []
    for token in _tokenize(title):
        if token[0].isupper() and token.lower() not in STOPWORDS:
            entities.append(token)
    return list(dict.fromkeys(entities))


def _classify_story(keywords: list[str], text: str) -> tuple[str, str, list[str], str]:
    for rule in ARCHETYPE_RULES:
        if rule["keywords"].intersection(set(keywords)):
            return (
                rule["archetype"],
                rule["category"],
                rule["motivations"],
                rule["moral"],
            )
    return "Narrativa factual", "conteúdo informativo", ["curiosidade"], "contexto prático"


def _infer_emotions(text: str) -> list[str]:
    matches = []
    lower_text = text.lower()
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(keyword in lower_text for keyword in keywords):
            matches.append(emotion)
    return matches


def _build_story_insights(
    title: str,
    description: str,
    keywords: list[str],
    entities: list[str],
    archetype: str,
    moral_angle: str,
) -> dict[str, str]:
    subject = entities[0] if entities else (keywords[0] if keywords else "o tema do vídeo")
    action = keywords[1] if len(keywords) > 1 else "o acontecimento central"
    tension = keywords[2] if len(keywords) > 2 else "o ponto de virada"

    return {
        "core_event": f"O vídeo relata como {subject} se conecta a {action}.",
        "narrative_tension": f"A tensão está em entender {tension} e as consequências diretas.",
        "emotional_pull": f"O público acompanha porque quer entender as implicações para {subject}.",
        "moral_angle": f"O eixo moral está em {moral_angle}.",
        "share_reason": f"As pessoas compartilham para discutir {archetype.lower()} e seus impactos.",
    }


def _build_similarity_reason(keywords: list[str], title: str) -> str:
    if not keywords or not title:
        return ""
    lowered_title = title.lower()
    matched = [kw for kw in keywords if kw in lowered_title]
    if matched:
        return f"Compartilha elementos sobre {', '.join(matched[:2])}."
    return ""


def _build_narrative_take(title: str, keywords: list[str]) -> str:
    if not title or not keywords:
        return ""
    lowered_title = title.lower()
    matched = [kw for kw in keywords if kw in lowered_title]
    if matched:
        return f"O título destaca {', '.join(matched[:2])} como eixo do conflito."
    return f"O título aponta para o conflito principal envolvendo {keywords[0]}."


def _build_opening_strategies(
    subject: str,
    action: str,
    story_insights: dict[str, str],
) -> list[str]:
    tension = story_insights.get("narrative_tension", "")

    strategies = [
        f"Abrir com o desfecho envolvendo {subject} e voltar para explicar {action}.",
        f"Começar no pico de tensão que envolve {subject} e retroceder para o gatilho.",
        f"Mostrar a reação imediata ao que aconteceu com {subject} antes de explicar o fato.",
    ]

    if tension:
        strategies.append(f"Destacar {tension.lower()} como cartão inicial antes da linha do tempo.")

    return strategies[:4]


def _build_sequence_preview(duration: int | None, platform: str) -> list[dict[str, Any]]:
    if not duration:
        return []

    limits = {
        "tt": (18, 30),
        "ig": (20, 30),
        "yt": (20, 35),
        "other": (18, 30),
    }
    ideal_min, ideal_max = limits.get(platform, (18, 30))

    roles = ["hook", "context"]
    if duration >= 90:
        roles.append("payoff")

    if duration < 30:
        roles = ["hook"]

    role_ranges = {
        "hook": (max(10, ideal_min), min(20, ideal_max)),
        "context": (max(20, ideal_min), min(35, ideal_max)),
        "payoff": (max(20, ideal_min), min(30, ideal_max)),
    }

    preview = []
    for role in roles:
        start, end = role_ranges[role]
        preview.append({
            "role": role,
            "duration": f\"{int(start)}–{int(end)}s\",
        })

    return preview


def _build_editorial_blueprint(subject: str, action: str, tension: str) -> dict[str, str]:
    return {
        "opening": f"Abrir com {subject} no momento imediatamente após {action}.",
        "setup": f"Apresentar quem é {subject} e o contexto que levou ao ponto de {action}.",
        "context": "Explicar os fatos anteriores em 2 ou 3 cenas curtas.",
        "tension": f"Concentrar a narrativa no que está em jogo e no impacto de {tension}.",
        "reveal": f"Mostrar a consequência direta do evento envolvendo {subject}.",
    }


def _build_candidate_decisions(
    keywords: list[str],
    entities: list[str],
    archetype: str,
    moral_angle: str,
    emotional_triggers: list[str],
) -> list[dict[str, Any]]:
    if not keywords and not entities:
        return []

    subject = entities[0] if entities else keywords[0]
    secondary = keywords[1] if len(keywords) > 1 else "o ponto crítico"
    tension = keywords[2] if len(keywords) > 2 else secondary

    candidates = [
        {
            "idea": f"Linha do tempo completa: de {subject} até o momento de {secondary}.",
            "performance_type": "Retention-driven",
            "risk_level": "low",
            "role": "timeline reconstruction",
            "trigger": "curiosidade organizada",
            "blueprint": _build_editorial_blueprint(subject, secondary, tension),
            "ending": "Cliffhanger",
            "rationale": "Organiza o enredo para quem chegou pelo corte e mantém a audiência acompanhando cada etapa.",
            "score": 3,
        },
        {
            "idea": f"Consequências imediatas: o que mudou para {subject} após {secondary}.",
            "performance_type": "Series-builder",
            "risk_level": "medium",
            "role": "context expansion",
            "trigger": "tensão e impacto social",
            "blueprint": _build_editorial_blueprint(subject, secondary, tension),
            "ending": "Part 2",
            "rationale": f"Foca no impacto humano de {secondary}, reforçando o ângulo de {moral_angle}.",
            "score": 2,
        },
    ]

    if archetype.startswith("Mistério") or "curiosidade" in emotional_triggers:
        candidates.append(
            {
                "idea": f"Pistas e lacunas: o que ainda não foi explicado sobre {subject}.",
                "performance_type": "Reach-driven",
                "risk_level": "high",
                "role": "part 2",
                "trigger": "mistério aberto",
                "blueprint": _build_editorial_blueprint(subject, secondary, tension),
                "ending": "Question",
                "rationale": "Explora o desconhecido e incentiva comentários com teorias específicas.",
                "score": 1,
            }
        )

    return candidates


def _rank_decisions(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_decisions = sorted(decisions, key=lambda d: d.get("score", 0), reverse=True)
    ranked = []
    for idx, decision in enumerate(sorted_decisions[:3], start=1):
        decision_copy = decision.copy()
        decision_copy["priority"] = idx
        ranked.append(decision_copy)
    return ranked


def _compute_confidence(
    keywords: list[str],
    emotional_triggers: list[str],
    similar_videos: list[dict[str, Any]],
    score: int,
) -> float:
    keyword_signal = min(len(keywords), 5) / 5
    emotion_signal = min(len(emotional_triggers), 3) / 3
    similarity_signal = min(len(similar_videos), 3) / 3
    score_signal = min(score, 100) / 100
    return round(
        (0.35 * score_signal)
        + (0.25 * similarity_signal)
        + (0.2 * keyword_signal)
        + (0.2 * emotion_signal),
        2,
    )


def build_analysis(url: str) -> ViralAnalysisResult:
    platform = detect_platform(url)
    metadata = {}

    try:
        metadata = fetch_metadata(url)
    except Exception:
        metadata = {}

    title = metadata.get("title") or ""
    description = metadata.get("description") or ""
    duration = metadata.get("duration") or 0
    views = metadata.get("views")
    likes = metadata.get("likes")
    comments = metadata.get("comments")

    engagement = calculate_engagement(views or 0, likes, comments)
    duration_score = score_duration(duration)
    engagement_score = score_engagement(engagement)

    keywords = _extract_keywords(f"{title} {description}")
    entities = _extract_entities(title)

    narrative_archetype, viral_category, motivations, moral_angle = _classify_story(
        keywords, f"{title} {description}"
    )
    emotional_triggers = list(dict.fromkeys(_infer_emotions(f"{title} {description}")))

    score = min(
        100,
        32 + duration_score * 12 + engagement_score * 18 + len(keywords) * 3,
    )

    if score >= 75:
        score_label = "Alto"
    elif score >= 55:
        score_label = "Médio"
    else:
        score_label = "Baixo"

    story_insights = _build_story_insights(
        title, description, keywords, entities, narrative_archetype, moral_angle
    )

    search_terms = " ".join((entities + keywords)[:5]).strip()
    similar_raw = search_similar_videos(search_terms, limit=6)
    similar_videos = []
    for item in similar_raw:
        if not item.get("title") or not item.get("url"):
            continue
        why_similar = _build_similarity_reason(keywords, item.get("title", ""))
        if not why_similar:
            continue
        narrative_take = _build_narrative_take(item.get("title", ""), keywords)
        similar_videos.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "views": item.get("views"),
                "thumbnail": item.get("thumbnail"),
                "platform": item.get("platform"),
                "why_similar": why_similar,
                "narrative_take": narrative_take,
            }
        )

    confidence_score = _compute_confidence(
        keywords=keywords,
        emotional_triggers=emotional_triggers,
        similar_videos=similar_videos,
        score=score,
    )
    confidence_label = "alta" if confidence_score >= 0.7 else "moderada"

    editorial_candidates = _build_candidate_decisions(
        keywords, entities, narrative_archetype, moral_angle, emotional_triggers
    )
    ranked_decisions = _rank_decisions(editorial_candidates)

    discarded_reason = None
    if confidence_score < 0.6 and ranked_decisions:
        ranked_decisions = ranked_decisions[:1]
        ranked_decisions[0]["recommendation_note"] = (
            "Única ideia recomendada (confiança moderada)."
        )
        discarded_reason = (
            "A confiança está abaixo do ideal; optamos por uma única direção clara."
        )
    elif len(ranked_decisions) < 2:
        discarded_reason = (
            "As informações disponíveis não sustentam múltiplos caminhos fortes; "
            "priorizamos a decisão mais segura."
        )

    if ranked_decisions:
        top = ranked_decisions[0]
        subject = entities[0] if entities else (keywords[0] if keywords else "o tema")
        action = keywords[1] if len(keywords) > 1 else "o ponto crítico"
        top["next_if_success"] = (
            f"Se performar bem, o próximo vídeo deve aprofundar como {subject} chegou a {action}."
        )
        top["opening_strategies"] = _build_opening_strategies(
            subject, action, story_insights
        )

    evidence_videos = similar_videos[:3]

    platform_key = {
        "youtube": "yt",
        "tiktok": "tt",
        "instagram": "ig",
    }.get(platform, "other")

    sequence_preview = _build_sequence_preview(
        duration=duration,
        platform=platform_key,
    )

    return ViralAnalysisResult(
        platform=platform,
        metadata={
            "title": title or "Título indisponível",
            "description": description or "Descrição indisponível",
            "duration": duration,
            "views": views,
            "likes": likes,
            "comments": comments,
            "engagement": round(engagement, 4),
            "thumbnail": metadata.get("thumbnail"),
        },
        score=score,
        score_label=score_label,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        narrative_archetype=narrative_archetype,
        viral_category=viral_category,
        audience_motivation=motivations,
        emotional_triggers=emotional_triggers or ["curiosidade"],
        story_insights=story_insights,
        similar_videos=similar_videos,
        evidence_videos=evidence_videos,
        editorial_decisions=ranked_decisions,
        sequence_preview=sequence_preview,
        discarded_reason=discarded_reason,
    )
