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
    narrative_archetype: str
    viral_category: str
    audience_motivation: list[str]
    emotional_triggers: list[str]
    story_insights: dict[str, str]
    content_cluster: list[str]
    related_videos: list[dict[str, str]]
    similar_videos: list[dict[str, Any]]
    opening_strategies: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "metadata": self.metadata,
            "score": self.score,
            "score_label": self.score_label,
            "narrative_archetype": self.narrative_archetype,
            "viral_category": self.viral_category,
            "audience_motivation": self.audience_motivation,
            "emotional_triggers": self.emotional_triggers,
            "story_insights": self.story_insights,
            "content_cluster": self.content_cluster,
            "related_videos": self.related_videos,
            "similar_videos": self.similar_videos,
            "opening_strategies": self.opening_strategies,
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


def _build_content_cluster(
    keywords: list[str],
    entities: list[str],
    tags: list[str],
    platform: str,
) -> list[str]:
    cluster: list[str] = []
    if not keywords and not entities:
        return cluster

    subject = entities[0] if entities else keywords[0]
    secondary = keywords[1] if len(keywords) > 1 else None

    cluster.append(f"Linha do tempo completa ligada a {subject}")
    if secondary:
        cluster.append(f"O ponto de virada: quando {subject} mudou por causa de {secondary}")
    if tags:
        cluster.append(f"Conexões com {tags[0]} e histórias parecidas")
    if platform == "tiktok":
        cluster.append(f"Versão rápida focada no pico de tensão de {subject}")
    if platform == "instagram":
        cluster.append(f"Série de Reels explicando a cronologia de {subject}")
    if platform == "youtube":
        cluster.append(f"Shorts sequenciais com capítulos do caso de {subject}")

    return cluster[:8]


def _build_related_videos(
    keywords: list[str],
    entities: list[str],
    archetype: str,
    moral_angle: str,
) -> list[dict[str, str]]:
    if not keywords and not entities:
        return []

    subject = entities[0] if entities else keywords[0]
    secondary = keywords[1] if len(keywords) > 1 else subject

    ideas = [
        {
            "idea": f"Reconstituir a sequência que levou {subject} ao ponto de {secondary}.",
            "focus": "timeline",
            "why": "Organiza a história para quem chegou pelo corte e melhora retenção.",
            "role": "timeline reconstruction",
        },
        {
            "idea": f"Mostrar o impacto imediato de {secondary} na vida de {subject}.",
            "focus": "consequência",
            "why": "Explora o conflito central e conecta emocionalmente a audiência.",
            "role": "context expansion",
        },
        {
            "idea": f"Apresentar versões conflitantes sobre {subject} e o que mudou depois.",
            "focus": "comparação",
            "why": "O contraste gera debate e prolonga o interesse da audiência.",
            "role": "comparison",
        },
    ]

    if archetype.startswith("Mistério"):
        ideas.append(
            {
                "idea": f"Mapear pistas que ainda deixam o caso de {subject} em aberto.",
                "focus": "lacunas",
                "why": "Mantém o suspense e incentiva comentários com teorias.",
                "role": "part 2",
            }
        )
    if archetype.startswith("Traição") or archetype.startswith("Engano"):
        ideas.append(
            {
                "idea": f"Mostrar como a quebra de confiança em {subject} mudou o entorno social.",
                "focus": "moral",
                "why": f"Explora o ângulo de {moral_angle} que faz o público reagir.",
                "role": "reaction",
            }
        )

    return ideas[:5]


def _build_opening_strategies(
    keywords: list[str],
    entities: list[str],
    story_insights: dict[str, str],
    similar_videos: list[dict[str, Any]],
) -> list[str]:
    if not keywords and not entities:
        return []

    subject = entities[0] if entities else keywords[0]
    action = keywords[1] if len(keywords) > 1 else "o ponto central"
    tension = story_insights.get("narrative_tension", "")

    strategies = [
        f"Abrir com o desfecho envolvendo {subject} e voltar para explicar {action}.",
        f"Começar no pico de tensão que envolve {subject} e retroceder para o gatilho.",
        f"Mostrar a reação imediata ao que aconteceu com {subject} antes de explicar o fato.",
    ]

    if similar_videos:
        top_similar = similar_videos[0].get("title")
        if top_similar:
            strategies.append(
                f"Referenciar o gancho narrativo de "
                f"“{top_similar[:60]}” e usar como comparação de ritmo." 
            )

    if tension:
        strategies.append(f"Destacar {tension.lower()} como cartão inicial antes da linha do tempo.")

    return strategies[:5]


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
    tags = metadata.get("tags") or []

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

    content_cluster = _build_content_cluster(keywords, entities, tags, platform)
    related_videos = _build_related_videos(keywords, entities, narrative_archetype, moral_angle)
    opening_strategies = _build_opening_strategies(
        keywords, entities, story_insights, similar_videos
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
        narrative_archetype=narrative_archetype,
        viral_category=viral_category,
        audience_motivation=motivations,
        emotional_triggers=emotional_triggers or ["curiosidade"],
        story_insights=story_insights,
        content_cluster=content_cluster,
        related_videos=related_videos,
        similar_videos=similar_videos,
        opening_strategies=opening_strategies,
    )
