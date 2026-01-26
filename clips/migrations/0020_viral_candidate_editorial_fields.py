from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("clips", "0019_storyclip_audio_path"),
    ]

    operations = [
        migrations.CreateModel(
            name="ViralCandidate",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("start_time", models.FloatField()),
                ("end_time", models.FloatField()),
                ("duration", models.FloatField()),
                ("transcript_text", models.TextField()),
                ("viral_score", models.IntegerField()),
                ("emotion", models.CharField(choices=[("curiosity", "Curiosity"), ("shock", "Shock"), ("opinion", "Opinion"), ("neutral", "Neutral")], default="neutral", max_length=16)),
                ("reason", models.TextField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("video_job", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="viral_candidates", to="clips.videojob")),
            ],
        ),
        migrations.AddField(
            model_name="videoclip",
            name="editorial_selected",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="videoclip",
            name="viral_candidate",
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="clips", to="clips.viralcandidate"),
        ),
    ]
