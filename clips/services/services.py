import gc
import os
import re
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yt_dlp
import pysubs2
from faster_whisper import WhisperModel
from yt_dlp import DownloadError

from analysis.services.viral_analysis import _extract_keywords
from subtitles.subtitle_builder import build_subtitle_filter
from clips.viral_engine.expansion import expand_window
from clips.viral_engine.profiles import get_profile
from clips.viral_engine.scoring import score_segment


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

MIN_GAP_BETWEEN_CLIPS = 60.0  # Evita clips muito grudados
def pick_viral_windows(transcript: dict, min_s=8, max_s=20, top_n=5, profile="podcast") -> list[dict]:
    """
    Motor Viral Pro:
    1. Pontua cada segmento usando hooks e arquétipos.
    2. Identifica 'âncoras' (momentos de pico).
    3. Expande a janela para trás (contexto) e para frente (conclusão).
    """
    segments = transcript.get("segments", [])
    if not segments:
        return []

    profile_data = get_profile(profile)

    # 1. Pontuar todos os segmentos
    scored_segments = []
    for i, seg in enumerate(segments):
        score, hooks = score_segment(seg, profile_data)
        scored_segments.append({
            "index": i,
            "start": seg["start"],
            "end": seg["end"],
            "score": score,
            "hooks": hooks,
            "text": seg["text"]
        })

    # 2. Rankear por score para achar as melhores âncoras
    anchors = sorted(scored_segments, key=lambda x: x["score"], reverse=True)

    picks = []
    for anchor in anchors:
        if len(picks) >= top_n:
            break

        # Verificar se esta âncora já está coberta por um clip selecionado
        # ou se está muito próxima de um clip existente
        is_too_close = False
        for p in picks:
            if abs(anchor["start"] - p["start"]) < MIN_GAP_BETWEEN_CLIPS:
                is_too_close = True
                break

        if is_too_close:
            continue

        # 3. Expandir a janela ao redor da âncora (Expansion Logic)
        # Passamos a âncora para garantir que o clip comece com o Hook
        window = expand_window(
            segments=scored_segments,
            anchor_idx=anchor["index"],
            min_s=min_s,
            max_s=max_s
        )

        if window:
            picks.append(window)

    # Ordenar por tempo de início
    picks.sort(key=lambda x: x["start"])
    return picks


def pick_viral_windows_rich(transcript: dict, min_s=18, max_s=40, top_k=6) -> list[dict]:
    segs = transcript.get("segments", [])
    if not segs:
        return []

    # 1) Texto global (para entender narrativa)
    full_text = " ".join((s.get("text") or "") for s in segs)
    keywords = _extract_keywords(full_text)
    archetype, category, motivations, moral_angle = _classify_story(keywords, full_text)
    emotions = _infer_emotions(full_text)

    # 2) hook words base + palavras derivadas do arquétipo
    base_hooks = {
        "curiosity": [
            "ninguém", "no one", "ninguém te conta", "segredo", "secret",
            "você não sabia", "you didn’t know", "sabia disso",
        ],
        "drama": [
            "traição", "vazou", "exposed", "exposição", "cancelado", "polêmica",
            "escândalo", "destruiu", "acabou",
        ],
        "money": [
            "dinheiro", "money", "ficar rico", "milionário", "fácil", "rápido",
            "erro que te impede", "perdendo dinheiro",
        ],
    }

    archetype_hooks = []
    if "Mistério" in archetype:
        archetype_hooks += ["mistério", "misterio", "sumiu", "desapareceu", "ninguém sabe"]
    if "Fraude" in archetype or "Engano" in archetype:
        archetype_hooks += ["golpe", "fraude", "esquema", "piramide", "furada"]
    if "Traição" in archetype:
        archetype_hooks += ["traiu", "traição", "vazou", "vazamento", "mentiu"]

    all_hooks = set(
        base_hooks["curiosity"]
        + base_hooks["drama"]
        + base_hooks["money"]
        + archetype_hooks
    )

    # 3) pré-cálculo segmento a segmento
    meta = []
    for s in segs:
        text = (s.get("text") or "").lower()
        tokens = re.findall(r"\w+", text)
        hooks = sum(1 for hw in all_hooks if hw.lower() in text)

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
        j = i
        while j < n and (meta[j]["end"] - meta[i]["start"]) < min_s:
            j += 1

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

            base_score = (tokens / max(dur, 1.0)) + hooks * 1.5

            # 4) boosts por emoção/narrativa
            boost = 1.0
            if "curiosidade" in emotions:
                boost += 0.15
            if "tensão" in emotions:
                boost += 0.10
            if "indignação" in emotions:
                boost += 0.10
            if archetype.startswith("Mistério"):
                boost += 0.15

            score = base_score * boost

            if best is None or score > best["score"]:
                best = {
                    "start": window[0]["start"],
                    "end": window[-1]["end"],
                    "score": float(score),
                    "caption": window[0]["text"][:120],
                }

            k += 1

        if best:
            picks.append(best)

        i += 1

    # mesmo filtro de overlap e top_k do MVP antigo
    picks.sort(key=lambda x: x["score"], reverse=True)

    filtered = []
    for p in picks:
        overlap = any(
            not (p["end"] <= f["start"] or p["start"] >= f["end"])
            for f in filtered
        )
        if not overlap:
            filtered.append(p)
        if len(filtered) >= top_k:
            break

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

def make_vertical_clip_with_captions(
    video_path: str,
    start: float,
    end: float,
    subtitle_path: str,
    media_root: Path,
    clip_id: str,
    output_path: Optional[Path] = None,  # <--- novo parâmetro opcional
    caption_style: str | None = None,
    caption_config: dict | None = None,
) -> tuple[str, str]:

    # diretório padrão, caso nenhum output_path seja passado
    default_out_dir = media_root / "videos" / "clips"

    if output_path is not None:
        out_mp4 = Path(output_path)
        # garante que o diretório do output_path existe
        ensure_dir(out_mp4.parent)
    else:
        ensure_dir(default_out_dir)
        out_mp4 = default_out_dir / f"{clip_id}.mp4"

    subtitle_filter = build_subtitle_filter(
        subtitle_path=subtitle_path,
        caption_style=caption_style,
        caption_config=caption_config,
    )

    vf_parts = [
        "setpts=PTS-STARTPTS",
        "scale=1080:1920:force_original_aspect_ratio=increase",
        "crop=1080:1920",
    ]
    if subtitle_filter:
        vf_parts.append(subtitle_filter)
    vf_parts.extend([
        "fps=30",
        "format=yuv420p",
    ])
    vf = ",".join(vf_parts)
    if subtitle_filter:
        print("[SUB] ✅ captions enabled")
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

    # caption simples (por enquanto vazio – ainda pode ser preenchido em quem chama)
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
    caption_style: str | None = None,
    caption_config: dict | None = None,
):
    subtitle_filter = build_subtitle_filter(
        subtitle_path=subtitle_path,
        caption_style=caption_style,
        caption_config=caption_config,
    )

    vf_parts = ["setpts=PTS-STARTPTS"]
    if crop:
        vf_parts.append(
            f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}"
        )
        vf_parts.append("scale=1080:1920")
    else:
        vf_parts.append("scale=1080:1920:force_original_aspect_ratio=increase")
        vf_parts.append("crop=1080:1920")
    if subtitle_filter:
        vf_parts.append(subtitle_filter)
    vf_parts.extend(["fps=30", "format=yuv420p"])
    vf = ",".join(vf_parts)
    if subtitle_filter:
        print("[SUB] ✅ captions enabled")
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
