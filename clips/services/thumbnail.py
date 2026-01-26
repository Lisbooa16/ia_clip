from __future__ import annotations

import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from django.conf import settings

from .services import FFMPEG_BIN

YOUTUBE_THUMB_MAXRES = "https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
YOUTUBE_THUMB_HQ = "https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


def generate_clip_thumbnail(
    video_path: str,
    timestamp: float,
    output_path: Path | None = None,
) -> str | None:
    if not video_path:
        return None

    if output_path is None:
        output_dir = Path(settings.MEDIA_ROOT) / "thumbnails"
        output_path = output_dir / f"thumb_{Path(video_path).stem}.jpg"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss",
        f"{max(timestamp, 0.0):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"[THUMB] Falha ao gerar thumbnail: {exc}")
        return None

    if not output_path.exists():
        return None
    return str(output_path)


def insert_thumbnail_as_first_frame(
    clip_video_path: str,
    youtube_url: str | None,
) -> bool:
    video_id = _extract_youtube_video_id(youtube_url or "")
    if not video_id:
        return False

    thumb_path = _download_youtube_thumbnail(video_id)
    if not thumb_path:
        return False

    clip_path = Path(clip_video_path)
    if not clip_path.exists():
        return False

    tmp_dir = clip_path.parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    vertical_thumb_path = tmp_dir / f"{clip_path.stem}_thumb_vertical.jpg"
    if not create_vertical_thumbnail_letterbox(thumb_path, vertical_thumb_path):
        return False
    intro_path = tmp_dir / f"{clip_path.stem}_thumb_intro.mp4"
    concat_path = tmp_dir / f"{clip_path.stem}_thumb_concat.mp4"

    intro_cmd = [
        FFMPEG_BIN,
        "-y",
        "-loop",
        "1",
        "-i",
        str(vertical_thumb_path),
        "-frames:v",
        "1",
        "-r",
        "30",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(intro_path),
    ]

    concat_cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(intro_path),
        "-i",
        str(clip_path),
        "-filter_complex",
        (
            "[0:v]fps=30,format=yuv420p,setpts=PTS-STARTPTS[v0];"
            "[1:v]fps=30,format=yuv420p,setpts=PTS-STARTPTS[v1];"
            "[v0][v1]concat=n=2:v=1:a=0[v]"
        ),
        "-map",
        "[v]",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(concat_path),
    ]

    try:
        subprocess.check_call(intro_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"[THUMB] Falha ao inserir thumbnail: {exc}")
        return False

    if not concat_path.exists():
        return False

    try:
        shutil.move(str(concat_path), str(clip_path))
    except Exception as exc:
        print(f"[THUMB] Falha ao substituir clip: {exc}")
        return False

    return True


def create_vertical_thumbnail_letterbox(
    input_thumb_path: Path,
    output_thumb_path: Path,
) -> bool:
    output_thumb_path = Path(output_thumb_path)
    output_thumb_path.parent.mkdir(parents=True, exist_ok=True)

    filter_complex = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,gblur=sigma=30[bg];"
        "[0:v]scale=1080:-1:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2"
    )

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(input_thumb_path),
        "-filter_complex",
        filter_complex,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_thumb_path),
    ]

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"[THUMB] Falha ao criar thumbnail vertical: {exc}")
        return False

    return output_thumb_path.exists()


def _download_youtube_thumbnail(video_id: str) -> Path | None:
    output_dir = Path(settings.MEDIA_ROOT) / "thumbnails"
    output_dir.mkdir(parents=True, exist_ok=True)

    maxres_path = output_dir / f"{video_id}_maxres.jpg"
    hq_path = output_dir / f"{video_id}_hq.jpg"

    if maxres_path.exists():
        return maxres_path
    if hq_path.exists():
        return hq_path

    for url, path in (
        (YOUTUBE_THUMB_MAXRES.format(video_id=video_id), maxres_path),
        (YOUTUBE_THUMB_HQ.format(video_id=video_id), hq_path),
    ):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                if resp.status != 200:
                    continue
                data = resp.read()
                if not data:
                    continue
                path.write_bytes(data)
                return path
        except urllib.error.URLError:
            continue
        except Exception:
            continue
    return None


def _extract_youtube_video_id(url: str) -> str | None:
    if not url:
        return None
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return None

    host = parsed.netloc.lower()
    if "youtu.be" in host:
        return parsed.path.strip("/").split("/")[0] or None
    if "youtube.com" in host or "youtube-nocookie.com" in host:
        qs = urllib.parse.parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            return qs["v"][0]
        parts = [p for p in parsed.path.split("/") if p]
        if "shorts" in parts:
            idx = parts.index("shorts")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        if "embed" in parts:
            idx = parts.index("embed")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return None
