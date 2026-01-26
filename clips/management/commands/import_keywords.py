from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from clips.services.keyword_importer import KeywordImporter


class Command(BaseCommand):
    help = "Importa keywords em lote (JSON, CSV, YAML ou TXT)."

    def add_arguments(self, parser):
        parser.add_argument("--file", required=True, help="Caminho do arquivo a importar.")
        parser.add_argument("--format", dest="format_hint", help="json|csv|yaml|txt")
        parser.add_argument("--category", help="Categoria default para TXT/CSV sem coluna category.")
        parser.add_argument("--language", default="pt", help="Idioma padrão (default: pt).")
        parser.add_argument("--dry-run", action="store_true", help="Valida sem gravar no banco.")

    def handle(self, *args, **options):
        path = Path(options["file"])
        importer = KeywordImporter(default_language=options["language"])
        result = importer.import_file(
            path=path,
            category=options.get("category"),
            dry_run=options.get("dry_run", False),
            format_hint=options.get("format_hint"),
        )
        errors = result.get("errors", [])
        stats = result.get("stats", {})

        if errors:
            for err in errors:
                self.stderr.write(f"Erro: {err}")

        self.stdout.write(
            "Resultado: "
            f"total={stats.get('total', 0)} "
            f"criados={stats.get('created', 0)} "
            f"ignorados={stats.get('ignored', 0)} "
            f"invalidos={stats.get('invalid', 0)}"
        )
