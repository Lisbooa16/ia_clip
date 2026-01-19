from pathlib import Path

from django.conf import settings
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from clips.models import ClipPublication

YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
TOKEN_URI = "https://oauth2.googleapis.com/token"


def _get_video_path(output_path: str) -> Path:
    path = Path(output_path)
    if not path.is_absolute():
        path = Path(settings.MEDIA_ROOT) / output_path
    return path


def _build_credentials() -> Credentials:
    youtube_cfg = settings.SOCIAL_PUBLISHING.get("youtube", {})
    refresh_token = youtube_cfg.get("refresh_token")
    if not refresh_token:
        raise ValueError("YOUTUBE_REFRESH_TOKEN não configurado.")
    creds = Credentials(
        None,
        refresh_token=refresh_token,
        token_uri=TOKEN_URI,
        client_id=youtube_cfg.get("client_id"),
        client_secret=youtube_cfg.get("client_secret"),
        scopes=YOUTUBE_SCOPES,
    )
    creds.refresh(Request())
    return creds


def upload_clip_publication(publication: ClipPublication) -> dict:
    video_path = _get_video_path(publication.clip.output_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {video_path}")

    creds = _build_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    privacy_status = settings.SOCIAL_PUBLISHING.get("youtube", {}).get(
        "privacy_status",
        "private",
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": publication.title,
                "description": publication.description,
                "categoryId": "22",
            },
            "status": {"privacyStatus": privacy_status},
        },
        media_body=MediaFileUpload(str(video_path), resumable=True),
    )

    response = None
    while response is None:
        _status, response = request.next_chunk()

    video_id = response.get("id")
    if not video_id:
        raise RuntimeError("Falha ao obter ID do vídeo no upload.")

    return {
        "video_id": video_id,
        "url": f"https://youtu.be/{video_id}",
    }