from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from django.db import transaction

from clips.models import KeywordSignal
from .keyword_signals import normalize_text


@dataclass
class ImportStats:
    total: int = 0
    created: int = 0
    ignored: int = 0
    invalid: int = 0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "created": self.created,
            "ignored": self.ignored,
            "invalid": self.invalid,
        }


class KeywordImporter:
    def __init__(self, default_language: str = "pt"):
        self.default_language = default_language

    def import_file(
        self,
        path: str | Path,
        category: str | None = None,
        dry_run: bool = False,
        format_hint: str | None = None,
    ) -> dict:
        path = Path(path)
        if not path.exists():
            return {"errors": [f"Arquivo não encontrado: {path}"], "stats": ImportStats().to_dict()}
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {"errors": ["Arquivo vazio."], "stats": ImportStats().to_dict()}

        fmt = (format_hint or path.suffix.lstrip(".") or "txt").lower()
        try:
            items = self._parse_payload(raw, fmt, category)
        except RuntimeError as exc:
            return {"errors": [str(exc)], "stats": ImportStats().to_dict()}
        return self.import_items(items, dry_run=dry_run)

    def import_items(self, items: Iterable[dict], dry_run: bool = False) -> dict:
        stats = ImportStats()
        errors: list[str] = []

        cleaned = []
        for idx, item in enumerate(items, 1):
            stats.total += 1
            payload = self._normalize_item(item)
            if not payload:
                stats.invalid += 1
                errors.append(f"Item inválido na linha {idx}.")
                continue
            cleaned.append(payload)

        if not cleaned:
            return {"errors": errors, "stats": stats.to_dict()}

        existing = self._load_existing_normalized(cleaned)
        to_create = []
        seen = set()
        for payload in cleaned:
            norm_key = (payload["category"], payload["language"], payload["normalized"])
            if norm_key in seen or norm_key in existing:
                stats.ignored += 1
                continue
            seen.add(norm_key)
            to_create.append(payload)

        if dry_run:
            stats.created = len(to_create)
            return {"errors": errors, "stats": stats.to_dict()}

        created = self._bulk_create(to_create)
        stats.created = created
        return {"errors": errors, "stats": stats.to_dict()}

    def _bulk_create(self, items: list[dict]) -> int:
        if not items:
            return 0
        rows = [
            KeywordSignal(
                category=item["category"],
                term=item["term"],
                language=item["language"],
                weight=item["weight"],
                is_active=item["is_active"],
            )
            for item in items
        ]
        with transaction.atomic():
            KeywordSignal.objects.bulk_create(rows, ignore_conflicts=True)
        return len(rows)

    def _load_existing_normalized(self, items: list[dict]) -> set[tuple[str, str, str]]:
        categories = {item["category"] for item in items}
        languages = {item["language"] for item in items}
        qs = KeywordSignal.objects.filter(category__in=categories, language__in=languages)
        existing = set()
        for row in qs:
            norm = normalize_text(row.term)
            existing.add((row.category, row.language, norm))
        return existing

    def _normalize_item(self, item: dict) -> dict | None:
        category = (item.get("category") or "").strip().lower()
        term = (item.get("term") or item.get("phrase") or "").strip()
        if not category or not term:
            return None
        language = (item.get("language") or self.default_language or "pt").strip().lower()
        weight = item.get("weight", 1)
        is_active = item.get("active", True)

        try:
            weight = int(weight)
        except (TypeError, ValueError):
            weight = 1
        is_active = bool(is_active)

        normalized = normalize_text(term)
        if not normalized:
            return None
        return {
            "category": category,
            "term": term,
            "language": language,
            "weight": weight,
            "is_active": is_active,
            "normalized": normalized,
        }

    def _parse_payload(self, raw: str, fmt: str, category: str | None) -> list[dict]:
        if fmt in {"json"}:
            return self._parse_json(raw)
        if fmt in {"csv"}:
            return self._parse_csv(raw, category)
        if fmt in {"yaml", "yml"}:
            return self._parse_yaml(raw)
        return self._parse_text(raw, category)

    def _parse_json(self, raw: str) -> list[dict]:
        payload = json.loads(raw)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            language = payload.get("language")
            if "categories" in payload and isinstance(payload["categories"], list):
                return self._expand_json_categories(payload["categories"], language)
            category = payload.get("category")
            items = payload.get("items", [])
            default_weight = payload.get("default_weight")
            default_active = payload.get("default_active")
            if isinstance(items, list):
                expanded = self._expand_item_list(
                    items,
                    category=category,
                    language=language,
                    default_weight=default_weight,
                    default_active=default_active,
                )
                return expanded
        return []

    def _expand_json_categories(self, categories: list, language: str | None) -> list[dict]:
        expanded: list[dict] = []
        for entry in categories:
            if not isinstance(entry, dict):
                continue
            category = entry.get("category")
            items = entry.get("items", [])
            default_weight = entry.get("default_weight")
            default_active = entry.get("default_active")
            expanded.extend(
                self._expand_item_list(
                    items,
                    category=category,
                    language=language or entry.get("language"),
                    default_weight=default_weight,
                    default_active=default_active,
                )
            )
        return expanded

    def _expand_item_list(
        self,
        items: list,
        category: str | None,
        language: str | None,
        default_weight: int | None,
        default_active: bool | None,
    ) -> list[dict]:
        expanded = []
        for item in items:
            if isinstance(item, str):
                payload = {"term": item}
            elif isinstance(item, dict):
                payload = dict(item)
            else:
                continue
            if category and "category" not in payload:
                payload["category"] = category
            if language and "language" not in payload:
                payload["language"] = language
            if default_weight is not None and "weight" not in payload:
                payload["weight"] = default_weight
            if default_active is not None and "active" not in payload:
                payload["active"] = default_active
            expanded.append(payload)
        return expanded

    def _parse_csv(self, raw: str, category: str | None) -> list[dict]:
        items = []
        reader = csv.DictReader(raw.splitlines())
        for row in reader:
            payload = {
                "category": row.get("category") or category,
                "term": row.get("term") or row.get("phrase"),
                "language": row.get("language"),
                "weight": row.get("weight"),
                "active": row.get("active"),
            }
            items.append(payload)
        return items

    def _parse_yaml(self, raw: str) -> list[dict]:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML não instalado.") from exc
        payload = yaml.safe_load(raw)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            category = payload.get("category")
            language = payload.get("language")
            items = payload.get("items", [])
            if isinstance(items, list):
                for item in items:
                    if category and "category" not in item:
                        item["category"] = category
                    if language and "language" not in item:
                        item["language"] = language
                return items
        return []

    def _parse_text(self, raw: str, category: str | None) -> list[dict]:
        if not category:
            return []
        items = []
        for line in raw.splitlines():
            term = line.strip()
            if not term:
                continue
            items.append({"category": category, "term": term})
        return items
