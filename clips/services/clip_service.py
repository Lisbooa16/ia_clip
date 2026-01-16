from clips.viral_engine.profiles import DEFAULT_PROFILE
from clips.viral_engine.window_picker import pick_viral_windows_generic


def generate_clips(transcript, profile=None):
    return pick_viral_windows_generic(
        transcript,
        profile or DEFAULT_PROFILE
    )
