from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("clips", "0021_videoclip_thumbnail_path"),
    ]

    operations = [
        migrations.AddField(
            model_name="videoclip",
            name="description",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="videoclip",
            name="viral_caption",
            field=models.TextField(blank=True),
        ),
    ]
