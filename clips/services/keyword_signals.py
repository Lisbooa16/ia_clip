from __future__ import annotations

import unicodedata
from functools import lru_cache

from clips.models import KeywordSignal


def normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(stripped.lower().split())


@lru_cache(maxsize=128)
def get_keywords_by_category(category: str, language: str = "pt") -> list[dict]:
    qs = KeywordSignal.objects.filter(
        category=category,
        language=language,
        is_active=True,
    ).order_by("term")
    return [
        {
            "term": row.term,
            "normalized": normalize_text(row.term),
            "weight": int(row.weight or 1),
        }
        for row in qs
    ]


@lru_cache(maxsize=1)
def get_all_keywords(language: str = "pt") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    qs = KeywordSignal.objects.filter(language=language, is_active=True).order_by("category", "term")
    for row in qs:
        out.setdefault(row.category, []).append(
            {
                "term": row.term,
                "normalized": normalize_text(row.term),
                "weight": int(row.weight or 1),
            }
        )
    return out


def match_keywords(
    text: str,
    categories: list[str] | None = None,
    language: str = "pt",
) -> dict[str, list[str]]:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return {}

    all_keywords = get_all_keywords(language=language)
    if categories:
        keywords = {c: all_keywords.get(c, []) for c in categories}
    else:
        keywords = all_keywords

    tokens = set(normalized_text.split())
    matches: dict[str, list[str]] = {}
    for category, items in keywords.items():
        found = []
        for item in items:
            term = item["normalized"]
            if not term:
                continue
            if " " in term:
                if term in normalized_text:
                    found.append(item["term"])
            else:
                if term in tokens:
                    found.append(item["term"])
        if found:
            matches[category] = found
    return matches
