from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import yt_dlp

from analysis.utils import calculate_engagement, score_duration, score_engagement


PLATFORM_HOSTS = {
    "youtube": ("youtube.com", "youtu.be"),
    "tiktok": ("tiktok.com",),
    "instagram": ("instagram.com",),
}

HOOK_KEYWORDS = {
    "pt": [
        "segredo",
        "você não vai acreditar",
        "ninguém te conta",
        "o erro",
        "antes de",
        "depois de",
        "pior",
        "melhor",
        "como",
        "por que",
        "top",
        "descubra",
    ],
    "en": [
        "secret",
        "you won't believe",
        "nobody tells you",
        "mistake",
        "before",
        "after",
        "worst",
        "best",
        "how",
        "why",
        "top",
        "discover",
    ],
}

EMOTION_KEYWORDS = [
    "surpresa",
    "chocante",
    "incrível",
    "polêmico",
    "viral",
    "emocionante",
    "absurdo",
    "insano",
    "surprising",
    "shocking",
    "amazing",
    "controversial",
    "emotional",
    "insane",
]


@dataclass
class ViralAnalysisResult:
    platform: str
    metadata: dict[str, Any]
    score: int
    score_label: str
    explanation: list[str]
    signals: list[str]
    suggestions: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "metadata": self.metadata,
            "score": self.score,
            "score_label": self.score_label,
            "explanation": self.explanation,
            "signals": self.signals,
            "suggestions": self.suggestions,
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


def _keyword_matches(text: str, keywords: list[str]) -> list[str]:
    matches = []
    lower_text = text.lower()
    for keyword in keywords:
        if keyword in lower_text:
            matches.append(keyword)
    return matches


def build_analysis(url: str) -> ViralAnalysisResult:
    platform = detect_platform(url)
    metadata = {}
    explanation = []
    signals = []

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

    hook_hits = _keyword_matches(title, HOOK_KEYWORDS["pt"]) + _keyword_matches(
        title, HOOK_KEYWORDS["en"]
    )
    emotion_hits = _keyword_matches(
        f"{title} {description}", EMOTION_KEYWORDS
    )

    if hook_hits:
        signals.append("Título com gatilhos de abertura")
    if emotion_hits:
        signals.append("Presença de termos emocionais")
    if duration <= 60:
        signals.append("Formato curto favorece viralização")
    if duration_score <= 1:
        signals.append("Duração longa para cortes rápidos")
    if engagement_score >= 2:
        signals.append("Engajamento acima do esperado")

    score = min(
        100,
        25 + duration_score * 15 + engagement_score * 20 + len(hook_hits) * 5 + len(emotion_hits) * 4,
    )

    if score >= 75:
        score_label = "Alto"
        explanation.append("O vídeo tem sinais fortes de potencial viral.")
    elif score >= 55:
        score_label = "Médio"
        explanation.append("O vídeo tem bom potencial, mas pode precisar de ajustes.")
    else:
        score_label = "Baixo"
        explanation.append("O potencial viral parece limitado com os sinais atuais.")

    if duration_score >= 3:
        explanation.append("Duração curta favorece retenção e compartilhamento.")
    elif duration_score == 2:
        explanation.append("Duração moderada ainda é viável para cortes rápidos.")
    else:
        explanation.append("Duração longa sugere dividir em cortes menores.")

    if engagement_score >= 2:
        explanation.append("Engajamento acima da média reforça tração inicial.")
    elif engagement_score == 1:
        explanation.append("Engajamento moderado indica espaço para otimizar hooks.")
    else:
        explanation.append("Engajamento baixo reduz a probabilidade de viralizar.")

    summary_tags = tags[:5]
    related_ideas = _suggest_related_ideas(title, summary_tags, platform)
    viral_themes = _suggest_themes(hook_hits, emotion_hits, platform)
    clip_angles = _suggest_clip_angles(platform)
    hooks = _suggest_hooks(platform)
    captions = _suggest_captions(title, platform)

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
        explanation=explanation,
        signals=signals,
        suggestions={
            "related_ideas": related_ideas,
            "viral_themes": viral_themes,
            "clip_angles": clip_angles,
            "hooks": hooks,
            "captions": captions,
        },
    )


def _suggest_related_ideas(title: str, tags: list[str], platform: str) -> list[str]:
    base = [
        "Resumo em 30 segundos com ponto alto",
        "Parte 2 com curiosidade ou bastidor",
        "Reação ou opinião rápida sobre o assunto",
    ]
    if title:
        base.insert(0, f"Versão curta destacando: {title[:60]}")
    if tags:
        base.append(f"Explorar tema relacionado a: {', '.join(tags)}")
    if platform == "tiktok":
        base.append("Adicionar desafio ou trend do momento")
    if platform == "instagram":
        base.append("Transformar em sequência de Reels com legendas grandes")
    if platform == "youtube":
        base.append("Criar Shorts com cortes em série")
    return base[:5]


def _suggest_themes(hook_hits: list[str], emotion_hits: list[str], platform: str) -> list[str]:
    themes = [
        "Dicas rápidas",
        "História pessoal",
        "Comparativo antes/depois",
    ]
    if hook_hits:
        themes.append("Gatilho de curiosidade no primeiro segundo")
    if emotion_hits:
        themes.append("Explorar surpresa ou emoção")
    if platform == "tiktok":
        themes.append("Uso de áudio ou efeito em alta")
    if platform == "instagram":
        themes.append("Legenda curta com CTA para salvar")
    if platform == "youtube":
        themes.append("Contexto rápido + gancho para parte 2")
    return themes[:5]


def _suggest_clip_angles(platform: str) -> list[str]:
    angles = [
        "Corte com a melhor frase no início",
        "Pergunta direta para abrir o vídeo",
        "Trecho com emoção ou virada de história",
    ]
    if platform == "tiktok":
        angles.append("Transição rápida nos primeiros 2 segundos")
    if platform == "instagram":
        angles.append("Texto grande com benefício claro")
    if platform == "youtube":
        angles.append("Resumo em tópicos com ritmo acelerado")
    return angles


def _suggest_hooks(platform: str) -> list[str]:
    hooks = [
        "Você já viu isso acontecer?",
        "Se eu soubesse disso antes...",
        "O maior erro que todo mundo comete",
    ]
    if platform == "tiktok":
        hooks.append("Olha isso até o final")
    if platform == "instagram":
        hooks.append("Salva isso para depois")
    if platform == "youtube":
        hooks.append("Parte 1 de 2")
    return hooks


def _suggest_captions(title: str, platform: str) -> list[str]:
    captions = [
        "Corta isso agora 🔥",
        "Isso mudou tudo pra mim",
        "Assista até o fim",
    ]
    if title:
        captions.insert(0, f"Trecho chave: {title[:50]}")
    if platform == "instagram":
        captions.append("Marca alguém que precisa ver")
    if platform == "tiktok":
        captions.append("Trend do dia ✅")
    if platform == "youtube":
        captions.append("Shorts com insight rápido")
    return captions[:5]
