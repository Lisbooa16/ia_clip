from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("clips", "0006_alter_videojob_status"),
    ]

    operations = [
        migrations.CreateModel(
            name="VideoJobStep",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("step_name", models.CharField(max_length=64)),
                ("status", models.CharField(choices=[("pending", "Pending"), ("running", "Running"), ("done", "Done"), ("failed", "Failed")], default="pending", max_length=16)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("duration_seconds", models.FloatField(blank=True, null=True)),
                ("message", models.TextField(blank=True)),
                ("job", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="steps", to="clips.videojob")),
            ],
            options={
                "unique_together": {("job", "step_name")},
            },
        ),
    ]
