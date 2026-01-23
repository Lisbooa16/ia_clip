from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("clips", "0014_videojob_processing_mode"),
    ]

    operations = [
        migrations.AddField(
            model_name="videoclip",
            name="original_video_path",
            field=models.CharField(blank=True, max_length=500),
        ),
        migrations.AddField(
            model_name="videoclip",
            name="original_start",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="videoclip",
            name="original_end",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="videoclip",
            name="edited_start",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="videoclip",
            name="edited_end",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="videoclip",
            name="caption_style",
            field=models.CharField(
                choices=[("static", "Static"), ("word_by_word", "Word by word")],
                default="static",
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name="videoclip",
            name="caption_config",
            field=models.JSONField(blank=True, null=True),
        ),
    ]
