from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("clips", "0015_videoclip_editing_caption_style"),
    ]

    operations = [
        migrations.AddField(
            model_name="videojob",
            name="job_type",
            field=models.CharField(
                choices=[("video", "Video"), ("story", "Story")],
                default="video",
                max_length=10,
            ),
        ),
        migrations.AddField(
            model_name="videojob",
            name="story_text",
            field=models.TextField(blank=True),
        ),
        migrations.CreateModel(
            name="StoryClip",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("part_number", models.IntegerField()),
                ("text", models.TextField()),
                ("video_path", models.CharField(blank=True, max_length=500)),
                ("duration_seconds", models.FloatField(blank=True, null=True)),
                ("status", models.CharField(choices=[("pending", "Pending"), ("processing", "Processing"), ("done", "Done"), ("error", "Error")], default="pending", max_length=20)),
                ("error", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("job", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="story_clips", to="clips.videojob")),
            ],
        ),
    ]
