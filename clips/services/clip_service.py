from clips.viral_engine.profiles import DEFAULT_PROFILE, get_profile
from clips.viral_engine.window_picker import pick_viral_windows_generic


def generate_clips(transcript, channel_type="youtube_shorts", archetype=None, top_k=6):
    profile = get_profile(channel_type=channel_type, archetype=archetype)
    picks = pick_viral_windows_generic(
        transcript,
        profile,
        min_s=40,
        max_s=60,
        top_k=top_k
    )

    return picks
