import random

BASE_TEMPLATES = [
    "-rank(delta(close, {w}))",
    "rank(ts_corr(close, volume, {w}))",
    "-rank(ts_rank(close, {w})) * rank(volume / adv{v})",
    "rank(returns(close, {w})) * -rank(ts_std(returns(close, 1), {v}))",
    "rank(delta(volume, {w})) * -rank(delta(close, {w}))",
]


def generate_templates(n=20, seed=42):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        t = rng.choice(BASE_TEMPLATES)
        out.append(t)
    return out
