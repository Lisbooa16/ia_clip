from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("clips", "0013_alter_clippublication_id"),
    ]

    operations = [
        migrations.AddField(
            model_name="videojob",
            name="processing_mode",
            field=models.CharField(
                choices=[("clips", "Clips"), ("full", "Full")],
                default="clips",
                max_length=10,
            ),
        ),
    ]
