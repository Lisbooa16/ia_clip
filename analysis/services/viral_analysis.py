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
        "emotions": ["mistério", "ansiedade"],
    },
    {
        "keywords": {"golpe", "fraude", "esquema", "pirâmide", "piramide"},
        "archetype": "Engano / Fraude",
        "category": "escândalo / denúncia",
        "motivations": ["choque", "alerta"],
        "emotions": ["indignação", "surpresa"],
    },
    {
        "keywords": {"traição", "traiu", "enganou", "vazou"},
        "archetype": "Traição / Ruptura",
        "category": "conflito pessoal",
        "motivations": ["curiosidade", "drama"],
        "emotions": ["raiva", "tristeza"],
    },
    {
        "keywords": {"prisão", "preso", "condenado", "tribunal", "crime"},
        "archetype": "Queda / Consequência",
        "category": "crime-adjacent",
        "motivations": ["curiosidade", "justiça"],
        "emotions": ["tensão", "choque"],
    },
    {
        "keywords": {"fuga", "escapou", "resgate", "capturado"},
        "archetype": "Escape / Sobrevivência",
        "category": "ação / sobrevivência",
        "motivations": ["tensão", "adrenalina"],
        "emotions": ["alívio", "medo"],
    },
    {
        "keywords": {"demissão", "demitido", "cancelado", "polêmica", "polêmica"},
        "archetype": "Queda Pública",
        "category": "controverso",
        "motivations": ["curiosidade", "conflito"],
        "emotions": ["indignação", "surpresa"],
    },
]

EMOTION_KEYWORDS = {
    "surpresa": {"surpresa", "chocante", "incrível", "inesperado", "absurdo"},
    "tensão": {"tensão", "mistério", "risco", "perigo", "ameaça"},
    "empatia": {"emocionante", "comovente", "história", "superação", "família"},
    "indignação": {"polêmica", "polêmica", "injustiça", "revolta", "traição"},
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
    story_analysis: dict[str, str]
    content_cluster: list[str]
    related_videos: list[dict[str, str]]
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
            "story_analysis": self.story_analysis,
            "content_cluster": self.content_cluster,
            "related_videos": self.related_videos,
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
    }


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


def _classify_story(keywords: list[str], text: str) -> tuple[str, str, list[str], list[str]]:
    for rule in ARCHETYPE_RULES:
        if rule["keywords"].intersection(set(keywords)):
            return (
                rule["archetype"],
                rule["category"],
                rule["motivations"],
                rule["emotions"],
            )
    emotional_triggers = _infer_emotions(text)
    return "Narrativa factual", "conteúdo informativo", ["curiosidade"], emotional_triggers


def _infer_emotions(text: str) -> list[str]:
    matches = []
    lower_text = text.lower()
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(keyword in lower_text for keyword in keywords):
            matches.append(emotion)
    return matches


def build_analysis(url: str) -> ViralAnalysisResult:
    platform = detect_platform(url)
    metadata = {}
    explanation = []

    try:
        metadata = fetch_metadata(url)
    except Exception:
        metadata = {}
        explanation.append("Não foi possível capturar todos os metadados agora.")

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

    narrative_archetype, viral_category, motivations, archetype_emotions = _classify_story(
        keywords, f"{title} {description}"
    )
    emotional_triggers = list(dict.fromkeys(archetype_emotions + _infer_emotions(description)))

    score = min(
        100,
        30 + duration_score * 12 + engagement_score * 18 + len(keywords) * 3,
    )

    if score >= 75:
        score_label = "Alto"
    elif score >= 55:
        score_label = "Médio"
    else:
        score_label = "Baixo"

    story_analysis = _build_story_analysis(title, description, keywords, entities)

    content_cluster = _build_content_cluster(keywords, entities, tags, platform)
    related_videos = _build_related_videos(keywords, entities, narrative_archetype, platform)
    opening_strategies = _build_opening_strategies(
        keywords, entities, narrative_archetype, story_analysis
    )

    if platform == "desconhecida":
        explanation.append("Plataforma não reconhecida, análise baseada em heurísticas gerais.")

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
        },
        score=score,
        score_label=score_label,
        narrative_archetype=narrative_archetype,
        viral_category=viral_category,
        audience_motivation=motivations,
        emotional_triggers=emotional_triggers or ["curiosidade"],
        story_analysis=story_analysis,
        content_cluster=content_cluster,
        related_videos=related_videos,
        opening_strategies=opening_strategies,
    )


def _build_story_analysis(
    title: str, description: str, keywords: list[str], entities: list[str]
) -> dict[str, str]:
    subject = entities[0] if entities else (keywords[0] if keywords else "o tema do vídeo")
    action = keywords[1] if len(keywords) > 1 else "o acontecimento central"
    tension = keywords[2] if len(keywords) > 2 else "o ponto de virada"

    return {
        "what_happened": f"O vídeo gira em torno de {subject} e {action}.",
        "why_intriguing": f"A narrativa chama atenção pelo impacto de {tension} e pelas consequências envolvidas.",
        "tension": f"A tensão está em entender como {subject} chegou a {action} e o que mudou depois.",
    }


def _build_content_cluster(
    keywords: list[str],
    entities: list[str],
    tags: list[str],
    platform: str,
) -> list[str]:
    cluster: list[str] = []
    subject = entities[0] if entities else (keywords[0] if keywords else "o tema")
    secondary = keywords[1] if len(keywords) > 1 else None

    if subject:
        cluster.append(f"Linha do tempo detalhada de {subject}")
    if secondary:
        cluster.append(f"O que levou {subject} a {secondary}")
    if keywords:
        cluster.append(f"Repercussões e reações imediatas sobre {keywords[0]}")
    if tags:
        cluster.append(f"Conexões com {tags[0]} e temas próximos")
    if platform == "tiktok":
        cluster.append(f"Versão em cortes rápidos destacando {subject}")
    if platform == "instagram":
        cluster.append(f"Série de Reels explicando o contexto de {subject}")
    if platform == "youtube":
        cluster.append(f"Shorts sequenciais com os principais momentos de {subject}")

    return cluster[:8]


def _build_related_videos(
    keywords: list[str],
    entities: list[str],
    archetype: str,
    platform: str,
) -> list[dict[str, str]]:
    if not keywords and not entities:
        return []

    subject = entities[0] if entities else keywords[0]
    secondary = keywords[1] if len(keywords) > 1 else subject

    ideas = [
        {
            "idea": f"Explicar o contexto imediato antes de {subject} entrar em cena.",
            "why": "O público entende rapidamente o ponto de partida e se envolve na história.",
            "type": "backstory",
        },
        {
            "idea": f"Reconstituir a sequência de eventos que leva a {secondary}.",
            "why": "Ajuda a organizar a narrativa e aumenta retenção em cortes rápidos.",
            "type": "timeline reconstruction",
        },
        {
            "idea": f"Mostrar as reações mais fortes após {subject}.",
            "why": "Explora a emoção coletiva que alimenta o compartilhamento.",
            "type": "reaction",
        },
    ]

    if archetype.startswith("Mistério"):
        ideas.append(
            {
                "idea": f"Detalhar pistas ou lacunas que ainda cercam {subject}.",
                "why": "Mistérios mantêm a audiência curiosa para próximos vídeos.",
                "type": "context expansion",
            }
        )
    if archetype.startswith("Traição") or archetype.startswith("Engano"):
        ideas.append(
            {
                "idea": f"Comparar versões conflitantes sobre {subject}.",
                "why": "O contraste alimenta debate e retenção.",
                "type": "comparison",
            }
        )

    if platform == "youtube":
        ideas.append(
            {
                "idea": f"Criar parte 2 com foco nas consequências de {subject}.",
                "why": "Sustenta uma série curta sem perder o público inicial.",
                "type": "part 2",
            }
        )

    return ideas[:6]


def _build_opening_strategies(
    keywords: list[str],
    entities: list[str],
    archetype: str,
    story_analysis: dict[str, str],
) -> list[str]:
    if not keywords and not entities:
        return []

    subject = entities[0] if entities else keywords[0]
    action = keywords[1] if len(keywords) > 1 else "o ponto central"
    tension = story_analysis.get("tension", "")

    strategies = [
        f"Abrir com o momento em que {subject} aparece e cortar para explicar {action}.",
        f"Começar mostrando a consequência de {action} e voltar para o início da história.",
        f"Introduzir a dúvida principal sobre {subject} antes de revelar o que aconteceu.",
    ]

    if "Mistério" in archetype:
        strategies.append("Iniciar com a última pista conhecida e reconstruir o antes.")
    if tension:
        strategies.append(f"Destacar {tension} em um corte curto e abrir espaço para o contexto.")

    return strategies[:5]
