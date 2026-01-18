from .hooks import HOOKS

def score_segment(segment, profile):
    score = 0.0
    active_hooks = []

    for hook_name, fn in HOOKS.items():
        try:
            if fn(segment):
                weight = profile.get(hook_name, 0)
                score += weight
                active_hooks.append(hook_name)
        except Exception:
            continue

    return score, active_hooks