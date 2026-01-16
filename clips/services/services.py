import gc
import os
import re
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import yt_dlp
import pysubs2
from faster_whisper import WhisperModel
from yt_dlp import DownloadError

def gpu_cleanup():
    try:
        import torch, gc, sys
        sys.stdout.flush()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print("GPU cleanup skipped:", e)

WHISPER_MODEL = None

def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        WHISPER_MODEL = WhisperModel(
            "small",
            device="cuda",
            compute_type="float16",
            cpu_threads=4,
            num_workers=1,
        )
    return WHISPER_MODEL

FFMPEG_BIN = r"C:\Users\Pichau\OneDrive\Área de Trabalho\ffmpeg\bin\ffmpeg.exe"



def detect_source(url: str) -> str:
    u = url.lower()
    if "youtube.com" in u or "youtu.be" in u:
        return "yt"
    if "tiktok.com" in u:
        return "tt"
    if "instagram.com" in u or "instagr.am" in u:
        return "ig"
    return "other"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def normalize_tiktok_url(url: str) -> str:
    if "tiktok.com" not in url:
        return url
    return url.split("?")[0]

def get_cookie_file(media_root: Path, source: str, url: str | None = None) -> Path | None:
    yt_cookie = media_root / "cookies" / "youtube.txt"

    if source == "yt":
        return yt_cookie if yt_cookie.exists() else None

    # 🔥 fallback: detecta por URL
    if url and ("youtube.com" in url or "youtu.be" in url):
        return yt_cookie if yt_cookie.exists() else None

    cookies = {
        "tt": media_root / "cookies" / "tiktok.txt",
        "ig": media_root / "cookies" / "instagram.txt",
    }

    path = cookies.get(source)
    return path if path and path.exists() else None

def _extract_with_browser(url, opts):
    opts = dict(opts)
    opts["cookiesfrombrowser"] = ("chrome",)
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=True)

def download_video(url: str, media_root: Path, source: str) -> tuple[str, str]:
    if source == "tt":
        url = normalize_tiktok_url(url)

    out_dir = media_root / "videos" / "original"
    ensure_dir(out_dir)

    base_opts = {
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0 Safari/537.36"
            ),
        },
    }

    def _extract(ydl_opts):
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=True)

    # 1) tenta sem cookies
    try:
        print("[DL] tentando sem cookie")
        info = _extract(base_opts)

    except DownloadError as e:
        # 2) tenta cookiefile (WINDOWS: use só isso)
        cookie_file = get_cookie_file(media_root, source, url)
        print("[DL] cookie_file =", cookie_file)

        if not cookie_file:
            raise DownloadError(
                f"{e}\n\n"
                f"YouTube exige login. Exporte cookies para "
                f"{media_root / 'cookies' / 'youtube.txt'}"
            )

        opts = dict(base_opts)
        opts["cookiefile"] = str(cookie_file)

        print("[DL] tentando com cookiefile")
        info = _extract(opts)

    # ✅ filepath REAL do yt-dlp (não chute)
    filepath = None
    if info.get("requested_downloads"):
        filepath = info["requested_downloads"][0].get("filepath")
    if not filepath:
        # fallback
        filepath = info.get("_filename") or info.get("filepath")

    if not filepath:
        # último fallback (muito raro)
        vid = info.get("id")
        ext = info.get("ext", "mp4")
        filepath = str(out_dir / f"{vid}.{ext}")

    path = Path(filepath)
    title = info.get("title") or ""
    print(f"[DL] baixou: {path}")

    # garantir mp4
    if path.suffix.lower() != ".mp4":
        mp4_path = path.with_suffix(".mp4")
        subprocess.check_call([
            FFMPEG_BIN, "-y",
            "-i", str(path),
            "-c:v", "libx264",
            "-c:a", "aac",
            str(mp4_path),
        ])
        path = mp4_path

    return str(path), title

def transcribe_with_words(video_path: str, language: str = "auto") -> dict:
    print("🎧 [WHISPER] Inicializando transcrição")
    print(f"📁 [WHISPER] Arquivo: {video_path}")
    print(f"🌍 [WHISPER] Language: {language}")

    t0 = time.time()

    print("🚀 [WHISPER] Carregando modelo (GPU)...")
    model = WhisperModel(
        "small",
        device="cuda",
        compute_type="float16"
    )
    print(f"✅ [WHISPER] Modelo carregado em {time.time() - t0:.2f}s")

    kw = {}
    if language and language != "auto":
        kw["language"] = language

    print("🧠 [WHISPER] Iniciando transcribe()")
    t1 = time.time()

    segments_iter, info = model.transcribe(
        video_path,
        word_timestamps=True,
        vad_filter=True,
        **kw
    )

    print("⏳ [WHISPER] transcribe() iniciado, aguardando primeiros segmentos...")

    segments = []
    last_log = time.time()

    for i, s in enumerate(segments_iter):
        now = time.time()

        # log a cada 2s ou a cada segmento
        if now - last_log > 2:
            print(
                f"🟢 [WHISPER] Segment {i} "
                f"({s.start:.2f}s → {s.end:.2f}s) | "
                f"tempo total: {now - t1:.1f}s"
            )
            last_log = now

        seg = {
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip(),
            "words": [],
        }

        if s.words:
            for w in s.words:
                seg["words"].append({
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": (w.word or "").strip(),
                })

        segments.append(seg)

    print(
        f"🏁 [WHISPER] Finalizado "
        f"({len(segments)} segmentos) "
        f"em {time.time() - t1:.2f}s"
    )

    return {"segments": segments}

def pick_viral_windows(transcript: dict, min_s=18, max_s=40, top_k=6) -> list[dict]:
    """
    Heurística simples (MVP):
      - cria janelas por blocos de segmentos
      - score = densidade de palavras + presença de "hook words"
    """
    hook_words = [
        "olha", "presta", "atenção", "ninguém", "segredo", "erro", "fácil", "rápido", "dinheiro",
        "listen", "look", "attention", "nobody", "secret", "mistake", "easy", "fast", "money",
    ]

    segs = transcript["segments"]
    if not segs:
        return []

    # Pré-cálculo: tokens e hook hits por segmento
    meta = []
    for s in segs:
        text = s["text"].lower()
        tokens = re.findall(r"\w+", text)
        hooks = sum(1 for hw in hook_words if hw in text)
        meta.append({
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "tokens": len(tokens),
            "hooks": hooks,
        })

    picks = []
    i = 0
    n = len(meta)

    while i < n:
        # abre janela a partir do i até atingir min_s
        j = i
        while j < n and (meta[j]["end"] - meta[i]["start"]) < min_s:
            j += 1

        # tenta expandir até max_s
        best = None
        k = j
        while k < n and (meta[k]["end"] - meta[i]["start"]) <= max_s:
            window = meta[i:k+1]
            dur = window[-1]["end"] - window[0]["start"]
            if dur < min_s:
                k += 1
                continue

            tokens = sum(x["tokens"] for x in window)
            hooks = sum(x["hooks"] for x in window)
            score = (tokens / max(dur, 1.0)) + hooks * 1.2  # densidade + hooks

            if best is None or score > best["score"]:
                best = {
                    "start": window[0]["start"],
                    "end": window[-1]["end"],
                    "score": float(score),
                    "caption": window[0]["text"][:120],  # caption inicial (você pode melhorar depois)
                }
            k += 1

        if best:
            picks.append(best)

        i += 1

    # remove sobreposições (simples) e pega top_k
    picks.sort(key=lambda x: x["score"], reverse=True)

    filtered = []
    for p in picks:
        overlap = False
        for f in filtered:
            if not (p["end"] <= f["start"] or p["start"] >= f["end"]):
                overlap = True
                break
        if not overlap:
            filtered.append(p)
        if len(filtered) >= top_k:
            break

    # ordena por tempo para ficar bonito
    filtered.sort(key=lambda x: x["start"])
    return filtered

def calc_font_size(text: str):
    """
    Fonte base estilo CapCut.
    Só diminui se a frase for grande.
    """
    length = len(text.replace("\\N", " "))

    BASE = 25  # tamanho padrão CapCut-like

    if length <= 16:
        return BASE
    if length <= 26:
        return BASE - 3
    if length <= 36:
        return BASE - 6
    return BASE - 9

def build_capcut_ass(words: list[dict], out_ass_path: str, font="Arial", font_size=25):
    """
    Legenda estilo CapCut (simples):
      - texto grande embaixo, stroke forte
      - gera linhas curtas por "frase" (agrupa palavras por ~1.6s ou pontuação)
    """
    subs = pysubs2.SSAFile()

    style = pysubs2.SSAStyle()
    style.fontname = font
    style.fontsize = font_size
    style.primarycolor = pysubs2.Color(255, 255, 255, 0)   # branco
    style.outlinecolor = pysubs2.Color(0, 0, 0, 0)         # preto
    style.backcolor = pysubs2.Color(0, 0, 0, 0)
    style.outline = 3
    style.shadow = 0
    style.alignment = 2  # bottom-center
    style.marginl = 120
    style.marginr = 120
    style.marginv = 40
    subs.styles["CapCut"] = style

    # agrupa palavras
    buf = []
    start = None
    last_end = None

    def smart_wrap(words, max_chars=10):
        lines = []
        current = ""

        for w in words:
            if len(current) + len(w) <= max_chars:
                current += (" " if current else "") + w
            else:
                lines.append(current)
                current = w

            if len(lines) == 2:
                break

        if current and len(lines) < 2:
            lines.append(current)

        return "\\N".join(lines)

    def flush():
        nonlocal buf, start, last_end
        if not buf or start is None or last_end is None:
            buf = []
            start = None
            last_end = None
            return

        text = smart_wrap(buf)
        font_size = calc_font_size(text)

        style_name = f"CapCut_{font_size}"

        if style_name not in subs.styles:
            s = pysubs2.SSAStyle()
            s.fontname = "Arial Black"
            s.fontsize = font_size
            s.primarycolor = pysubs2.Color(255, 255, 255, 0)
            s.outlinecolor = pysubs2.Color(0, 0, 0, 0)
            s.outline = max(2, font_size // 14)
            s.shadow = 0
            s.alignment = 2
            s.marginl = 40
            s.marginr = 40
            s.marginv = 40
            subs.styles[style_name] = s

        ev = pysubs2.SSAEvent(
            start=int(start * 1000),
            end=int(last_end * 1000),
            text=text,
            style=style_name,
        )

        subs.events.append(ev)
        buf = []
        start = None
        last_end = None

    for w in words:
        word = (w["word"] or "").strip()
        if not word:
            continue

        if start is None:
            start = w["start"]
        last_end = w["end"]
        buf.append(word)

        dur = last_end - start
        if dur >= 1.6 or re.search(r"[.!?]$", word):
            flush()

    flush()
    subs.save(out_ass_path)

def build_words_timeline(transcript, clip_start, clip_end, max_gap=0.5):
    words = []

    # coleta palavras reais
    for s in transcript["segments"]:
        for w in s.get("words", []):
            if w["end"] <= clip_start:
                continue
            if w["start"] >= clip_end:
                continue

            words.append({
                "start": max(0.0, w["start"] - clip_start),
                "end": min(clip_end, w["end"]) - clip_start,
                "word": w["word"] or "…",
            })

    if not words:
        # fallback total (vídeo nunca fica sem legenda)
        return [{
            "start": 0.0,
            "end": clip_end - clip_start,
            "word": "…",
        }]

    # ordena
    words.sort(key=lambda w: w["start"])

    # preenche gaps
    filled = []
    last_end = 0.0

    for w in words:
        if w["start"] - last_end > max_gap:
            filled.append({
                "start": last_end,
                "end": w["start"],
                "word": "…",
            })
        filled.append(w)
        last_end = w["end"]

    # cobre final
    total_dur = clip_end - clip_start
    if total_dur - last_end > max_gap:
        filled.append({
            "start": last_end,
            "end": total_dur,
            "word": "…",
        })

    return filled

def make_vertical_clip_with_captions(
    video_path: str,
    start: float,
    end: float,
    subtitle_path: str,
    media_root: Path,
    clip_id: str,
) -> tuple[str, str]:

    out_dir = media_root / "videos" / "clips"
    ensure_dir(out_dir)

    out_mp4 = out_dir / f"{clip_id}.mp4"

    sub_path = subtitle_path.replace("\\", "/").replace(":", "\\:")

    font_size = 44
    margin_v = 140
    margin_h = 64
    font_size = max(36, min(font_size, 48))
    margin_v = min(margin_v, 180)
    subtitle_style = (
        "FontName=Montserrat,"
        f"FontSize={font_size},"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BackColour=&H00000000,"
        "Bold=1,"
        "Outline=2,"
        "Shadow=1,"
        "Alignment=2,"
        f"MarginL={margin_h},"
        f"MarginR={margin_h},"
        f"MarginV={margin_v},"
        "PlayResX=1080,"
        "PlayResY=1920,"
        "WrapStyle=2"
    )
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        f"subtitles=filename='{sub_path}':force_style='{subtitle_style}',"
        "fps=30,format=yuv420p"
    )
    print(
        "[SUB] ✅ style=shortform "
        f"font=Montserrat size={font_size} margin_v={margin_v} res=1080x1920"
    )
    print("[RENDER] 🎞️ output_fps=30")

    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", video_path,
        "-vf", vf,
        "-r", "30",
        "-fps_mode", "cfr",
        "-avoid_negative_ts", "make_zero",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "128k",
        str(out_mp4),
    ]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        fallback_cmd = cmd[:]
        fallback_cmd[fallback_cmd.index("-preset") + 1] = "ultrafast"
        fallback_cmd[fallback_cmd.index("-crf") + 1] = "23"
        print("[RENDER] ⚠️ fallback render retry")
        subprocess.check_call(fallback_cmd)

    # caption simples (opcional, pode vir de outro lugar)
    caption = ""

    return str(out_mp4), caption


def make_vertical_clip_with_focus(
    video_path: str,
    start: float,
    end: float,
    subtitle_path: str,
    media_root: Path,
    clip_id: str,
    crop: dict | None,
    output_path: Path,
):
    sub_path = subtitle_path.replace("\\", "/").replace(":", "\\:")
    font_size = 44
    margin_v = 140
    margin_h = 64
    font_size = max(36, min(font_size, 48))
    margin_v = min(margin_v, 180)
    subtitle_style = (
        "FontName=Montserrat,"
        f"FontSize={font_size},"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BackColour=&H00000000,"
        "Bold=1,"
        "Outline=2,"
        "Shadow=1,"
        "Alignment=2,"
        f"MarginL={margin_h},"
        f"MarginR={margin_h},"
        f"MarginV={margin_v},"
        "PlayResX=1080,"
        "PlayResY=1920,"
        "WrapStyle=2"
    )

    if crop:
        vf = (
            f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']},"
            "scale=1080:1920,"
            f"subtitles=filename='{sub_path}':force_style='{subtitle_style}',"
            "fps=30,format=yuv420p"
        )
    else:
        vf = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            f"subtitles=filename='{sub_path}':force_style='{subtitle_style}',"
            "fps=30,format=yuv420p"
        )
    print(
        "[SUB] ✅ style=shortform "
        f"font=Montserrat size={font_size} margin_v={margin_v} res=1080x1920"
    )
    print("[RENDER] 🎞️ output_fps=30")

    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", video_path,
        "-vf", vf,
        "-r", "30",
        "-fps_mode", "cfr",
        "-avoid_negative_ts", "make_zero",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path),
    ]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        fallback_cmd = cmd[:]
        fallback_cmd[fallback_cmd.index("-preset") + 1] = "ultrafast"
        fallback_cmd[fallback_cmd.index("-crf") + 1] = "23"
        print("[RENDER] ⚠️ fallback render retry")
        subprocess.check_call(fallback_cmd)


def trim_clip(
    video_path: str,
    start: float,
    end: float,
    output_path: Path,
):
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", video_path,
        "-vf", "setpts=PTS-STARTPTS",
        "-af", "aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS",
        "-r", "30",
        "-fps_mode", "cfr",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path),
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        fallback_cmd = cmd[:]
        fallback_cmd[fallback_cmd.index("-preset") + 1] = "ultrafast"
        fallback_cmd[fallback_cmd.index("-crf") + 1] = "23"
        print("[RENDER] ⚠️ trim fallback retry")
        subprocess.check_call(fallback_cmd)


def burn_subtitles(
    video_path: str,
    subtitle_path: str,
    output_path: Path,
):
    sub_path = subtitle_path.replace("\\", "/").replace(":", "\\:")
    font_size = 44
    margin_v = 140
    margin_h = 64
    font_size = max(36, min(font_size, 48))
    margin_v = min(margin_v, 180)
    subtitle_style = (
        "FontName=Montserrat,"
        f"FontSize={font_size},"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BackColour=&H00000000,"
        "Bold=1,"
        "Outline=2,"
        "Shadow=1,"
        "Alignment=2,"
        f"MarginL={margin_h},"
        f"MarginR={margin_h},"
        f"MarginV={margin_v},"
        "PlayResX=1080,"
        "PlayResY=1920,"
        "WrapStyle=2"
    )
    vf = (
        f"subtitles=filename='{sub_path}':force_style='{subtitle_style}',"
        "fps=30,format=yuv420p"
    )
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", video_path,
        "-vf", vf,
        "-r", "30",
        "-fps_mode", "cfr",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path),
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        fallback_cmd = cmd[:]
        fallback_cmd[fallback_cmd.index("-preset") + 1] = "ultrafast"
        fallback_cmd[fallback_cmd.index("-crf") + 1] = "23"
        print("[RENDER] ⚠️ subtitle burn fallback retry")
        subprocess.check_call(fallback_cmd)

def transcribe_with_words_to_file(
    video_path: str,
    output_path: str,
    language="auto",
    modelo=None,
    whisper_model=None,
    use_vad=True,
):
    from faster_whisper import WhisperModel
    import json
    import gc

    try:
        import torch
    except ImportError:
        torch = None

    # 🔥 reutiliza modelo
    whisper_model = get_whisper_model()

    kw = {
        "word_timestamps": True,
        "vad_filter": use_vad,
    }

    if language != "auto":
        kw["language"] = language

    segments_iter, _ = whisper_model.transcribe(video_path, **kw)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('{"segments":[\n')

        first = True
        count = 0

        for s in segments_iter:
            seg = {
                "start": float(s.start),
                "end": float(s.end),
                "text": (s.text or "").strip(),
                "words": [
                    {
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": (w.word or "").strip(),
                    }
                    for w in (s.words or [])
                ],
            }

            if not first:
                f.write(",\n")

            json.dump(seg, f, ensure_ascii=False)
            first = False
            count += 1

            if count % 20 == 0:
                print(f"[WHISPER] {count} segmentos gravados")

        f.write("\n]}")

    if torch:
        torch.cuda.empty_cache()
    gc.collect()

    print(f"[WHISPER] Finalizado ({count} segmentos)")
    return True
