from __future__ import annotations


def calculate_engagement(views: int, likes: int | None, comments: int | None) -> float:
    if views == 0:
        return 0.0
    return ((likes or 0) + (comments or 0)) / views


def score_duration(duration: int | None) -> int:
    if not duration:
        return 1
    if duration <= 60:
        return 4
    if duration <= 120:
        return 3
    if duration <= 180:
        return 2
    return 1


def score_engagement(engagement: float) -> int:
    if engagement >= 0.05:
        return 3
    if engagement >= 0.02:
        return 2
    if engagement >= 0.01:
        return 1
    return 0
