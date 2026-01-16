def calculate_engagement(views, likes, comments):
    if views == 0:
        return 0
    return ((likes or 0) + (comments or 0)) / views


def calculate_viral_score(views, engagement, duration):
    score = 0

    if views > 100_000:
        score += 3
    elif views > 10_000:
        score += 2
    elif views > 1_000:
        score += 1

    if engagement > 0.05:
        score += 3
    elif engagement > 0.02:
        score += 2
    elif engagement > 0.01:
        score += 1

    if duration and duration > 600:
        score += 1  # bom pra cortes

    return score
