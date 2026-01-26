from django.db import migrations, models


def seed_keyword_signals(apps, schema_editor):
    KeywordSignal = apps.get_model("clips", "KeywordSignal")

    seed = {
        "entity": {
            "youtube",
            "tiktok",
            "instagram",
            "brasil",
            "governo",
            "banco",
            "startup",
            "podcast",
            "jornal",
            "policia",
            "polícia",
        },
        "emotion": {
            "incrivel",
            "incrível",
            "absurdo",
            "chocante",
            "surreal",
            "assustador",
            "emocionante",
            "injusto",
            "triste",
            "furioso",
            "revoltante",
        },
        "opinion": {
            "acho",
            "penso",
            "opinião",
            "opinao",
            "discordo",
            "concordo",
            "errado",
            "certo",
            "absurdo",
            "polêmica",
            "polêmico",
            "controverso",
            "mentira",
            "verdade",
        },
        "curiosity": {
            "por que",
            "porque",
            "como",
            "segredo",
            "misterio",
            "mistério",
            "ninguem",
            "ninguém",
            "sabe",
            "revelou",
        },
        "shock": {
            "chocante",
            "absurdo",
            "surreal",
            "inacreditavel",
            "inacreditável",
            "bizarro",
        },
    }

    for category, terms in seed.items():
        for term in sorted(terms):
            KeywordSignal.objects.get_or_create(
                category=category,
                term=term,
                language="pt",
                defaults={"weight": 1, "is_active": True},
            )


def unseed_keyword_signals(apps, schema_editor):
    KeywordSignal = apps.get_model("clips", "KeywordSignal")
    KeywordSignal.objects.filter(language="pt").delete()


class Migration(migrations.Migration):

    dependencies = [
        ("clips", "0022_videoclip_copy_fields"),
    ]

    operations = [
        migrations.CreateModel(
            name="KeywordSignal",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("category", models.CharField(choices=[("entity", "Entity"), ("emotion", "Emotion"), ("opinion", "Opinion"), ("curiosity", "Curiosity"), ("shock", "Shock")], max_length=32)),
                ("term", models.CharField(max_length=120)),
                ("language", models.CharField(default="pt", max_length=10)),
                ("weight", models.IntegerField(default=1)),
                ("is_active", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "unique_together": {("category", "term", "language")},
            },
        ),
        migrations.RunPython(seed_keyword_signals, unseed_keyword_signals),
    ]
