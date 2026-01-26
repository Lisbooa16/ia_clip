from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("clips", "0018_alter_storyclip_id_alter_storyclippublication_id"),
    ]

    operations = [
        migrations.AddField(
            model_name="storyclip",
            name="audio_path",
            field=models.CharField(blank=True, max_length=500),
        ),
    ]
