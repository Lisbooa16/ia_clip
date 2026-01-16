# analysis/services/youtube.py
import yt_dlp
from datetime import datetime, timedelta

def search_youtube_videos(query: str, limit=10):
    one_week_ago = datetime.utcnow() - timedelta(days=7)

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "dump_single_json": True,
        "skip_download": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        data = ydl.extract_info(
            f"ytsearch{limit}:{query}",
            download=False
        )

    videos = []
    for e in data.get("entries", []):
        if not e:
            continue

        videos.append({
            "video_id": e.get("id"),
            "title": e.get("title"),
            "url": e.get("url"),
            "views": e.get("view_count", 0),
            "duration": e.get("duration", 0),
            "published_at": e.get("upload_date"),
        })

    return videos
