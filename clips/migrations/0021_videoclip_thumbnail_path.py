from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("clips", "0020_viral_candidate_editorial_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="videoclip",
            name="thumbnail_path",
            field=models.CharField(blank=True, max_length=500),
        ),
    ]
