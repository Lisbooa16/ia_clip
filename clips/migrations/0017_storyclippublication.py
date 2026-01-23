from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("clips", "0016_videojob_story_fields_storyclip"),
    ]

    operations = [
        migrations.CreateModel(
            name="StoryClipPublication",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("platform", models.CharField(choices=[("youtube", "YouTube")], max_length=20)),
                ("title", models.CharField(max_length=255)),
                ("description", models.TextField(blank=True)),
                ("status", models.CharField(choices=[("queued", "Queued"), ("publishing", "Publishing"), ("published", "Published"), ("error", "Error")], default="queued", max_length=20)),
                ("external_url", models.URLField(blank=True)),
                ("error", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("clip", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="publications", to="clips.storyclip")),
            ],
        ),
    ]
