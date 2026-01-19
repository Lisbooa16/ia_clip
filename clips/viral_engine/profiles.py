# Perfis baseados em heurísticas estruturais + hooks virais
DEFAULT_PROFILE = {
    # heurísticas estruturais
    "question": 2.0,
    "short_sentence": 1.5,
    "pause": 1.0,
    "negation": 1.2,
    "numbers": 0.8,

    # hooks virais
    "curiosity": 2.5,
    "drama": 2.2,
    "money": 1.8,
    "fraud": 2.0,
    "urgency": 1.5,
    "social_proof": 1.3,
    "clickbait": 2.0,
}

CHANNEL_PROFILES = {
    "youtube_shorts": {
        **DEFAULT_PROFILE,
        "question": 2.5,
        "short_sentence": 1.8,
        "pause": 1.2,
        "curiosity": 3.0,
        "urgency": 2.0,
        "clickbait": 2.5,
    },
    "tiktok": {
        **DEFAULT_PROFILE,
        "short_sentence": 2.2,
        "question": 2.0,
        "drama": 2.8,
        "social_proof": 2.0,
    },
    "podcast": {
        **DEFAULT_PROFILE,
        "short_sentence": 1.2,
        "question": 1.8,
        "curiosity": 2.0,
        "money": 1.5,
        "pause": 0.6,
    },
}


ARCHETYPE_PROFILES = {
    "Mistério": {
        "curiosity": 3.5,
        "drama": 2.0,
    },
    "Fraude": {
        "fraud": 3.2,
        "money": 2.5,
        "urgency": 2.2,
    },
    "Traição": {
        "drama": 3.5,
        "curiosity": 2.0,
    },
    "Crime": {
        "fraud": 2.5,
        "drama": 2.5,
        "curiosity": 2.0,
    },
}


def get_profile(channel_type: str | None = None, archetype: str | None = None) -> dict:
    """
    Monta um profile final com base em:
    - channel_type: 'youtube_shorts', 'tiktok', 'podcast', etc.
    - archetype: 'Mistério', 'Fraude', etc. (opcional)
    """
    # base
    profile = DEFAULT_PROFILE.copy()

    # override por canal
    if channel_type and channel_type in CHANNEL_PROFILES:
        chan = CHANNEL_PROFILES[channel_type]
        profile.update(chan)

    # boost por arquétipo
    if archetype and archetype in ARCHETYPE_PROFILES:
        arch = ARCHETYPE_PROFILES[archetype]
        profile.update({**profile, **arch})

    return profile
