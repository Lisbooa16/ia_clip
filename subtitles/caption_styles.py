from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CaptionStyle(str, Enum):
    STATIC = "static"
    WORD_BY_WORD = "word_by_word"


class CaptionPosition(str, Enum):
    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"


@dataclass(frozen=True)
class CaptionStyleConfig:
    font_family: str = "Montserrat"
    font_size: int = 44
    font_color: str = "#FFFFFF"
    highlight_color: str = "#FFD84D"
    background: bool = True
    position: CaptionPosition = CaptionPosition.BOTTOM


DEFAULT_CAPTION_CONFIG = CaptionStyleConfig()


def normalize_style(value: str | None) -> CaptionStyle:
    if not value:
        return CaptionStyle.STATIC
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {CaptionStyle.WORD_BY_WORD.value, "karaoke"}:
        return CaptionStyle.WORD_BY_WORD
    return CaptionStyle.STATIC


def normalize_position(value: str | CaptionPosition | None) -> CaptionPosition:
    if isinstance(value, CaptionPosition):
        return value
    if not value:
        return DEFAULT_CAPTION_CONFIG.position
    normalized = str(value).strip().lower()
    if normalized in {"top", "upper"}:
        return CaptionPosition.TOP
    if normalized in {"center", "middle"}:
        return CaptionPosition.CENTER
    return CaptionPosition.BOTTOM


def normalize_config(data: dict | None) -> CaptionStyleConfig:
    if not data:
        return DEFAULT_CAPTION_CONFIG
    return CaptionStyleConfig(
        font_family=str(data.get("font_family", DEFAULT_CAPTION_CONFIG.font_family)),
        font_size=int(data.get("font_size", DEFAULT_CAPTION_CONFIG.font_size)),
        font_color=str(data.get("font_color", DEFAULT_CAPTION_CONFIG.font_color)),
        highlight_color=str(data.get("highlight_color", DEFAULT_CAPTION_CONFIG.highlight_color)),
        background=bool(data.get("background", DEFAULT_CAPTION_CONFIG.background)),
        position=normalize_position(data.get("position", DEFAULT_CAPTION_CONFIG.position)),
    )


def _normalize_hex_color(value: str) -> str:
    if not value:
        return "FFFFFF"
    raw = value.strip()
    if raw.startswith("&H"):
        return raw
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        return "FFFFFF"
    return raw.upper()


def to_ass_color(value: str, alpha: str = "00") -> str:
    normalized = _normalize_hex_color(value)
    if normalized.startswith("&H"):
        return normalized
    rr = normalized[0:2]
    gg = normalized[2:4]
    bb = normalized[4:6]
    return f"&H{alpha}{bb}{gg}{rr}"


def alignment_for_position(position: CaptionPosition) -> int:
    if position == CaptionPosition.TOP:
        return 8
    if position == CaptionPosition.CENTER:
        return 5
    return 2


def margins_for_position(position: CaptionPosition) -> tuple[int, int, int]:
    margin_l = 64
    margin_r = 64
    if position == CaptionPosition.TOP:
        margin_v = 120
    elif position == CaptionPosition.CENTER:
        margin_v = 80
    else:
        margin_v = 140
    return margin_l, margin_r, margin_v


def build_force_style(config: CaptionStyleConfig, play_res: tuple[int, int]) -> str:
    margin_l, margin_r, margin_v = margins_for_position(config.position)
    border_style = 3 if config.background else 1
    back_alpha = "80" if config.background else "00"
    return (
        f"FontName={config.font_family},"
        f"FontSize={max(24, int(config.font_size))},"
        f"PrimaryColour={to_ass_color(config.font_color)},"
        "OutlineColour=&H00000000,"
        f"BackColour={to_ass_color('#000000', back_alpha)},"
        "Bold=1,"
        "Outline=2,"
        "Shadow=1,"
        f"Alignment={alignment_for_position(config.position)},"
        f"MarginL={margin_l},"
        f"MarginR={margin_r},"
        f"MarginV={margin_v},"
        f"PlayResX={play_res[0]},"
        f"PlayResY={play_res[1]},"
        f"BorderStyle={border_style},"
        "WrapStyle=2"
    )
