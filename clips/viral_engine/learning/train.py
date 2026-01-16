from sklearn.linear_model import Ridge
import numpy as np
from viral_engine.hooks import HOOKS

def learn_profile(feedbacks):
    hook_names = list(HOOKS.keys())
    X, y = [], []

    for fb in feedbacks:
        X.append([1 if h in fb.hooks else 0 for h in hook_names])
        y.append(fb.retention_50p)

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    return dict(zip(hook_names, model.coef_))
