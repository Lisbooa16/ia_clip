from django.contrib import admin

from clips.models import KeywordSignal, VideoClip, VideoJob

admin.site.register(VideoJob)
admin.site.register(VideoClip)
admin.site.register(KeywordSignal)
