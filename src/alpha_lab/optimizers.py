import itertools
import random

import numpy as np


def expand_grid(param_space: dict):
    keys = list(param_space)
    vals = []
    for k in keys:
        v = param_space[k]
        if isinstance(v, tuple) and len(v) == 2:
            vals.append(np.linspace(float(v[0]), float(v[1]), 7).tolist())
        else:
            vals.append(list(v))
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo, strict=False))


def sample_random(param_space: dict, n: int, seed: int = 42):
    rng = random.Random(seed)
    keys = list(param_space)
    for _ in range(int(n)):
        p = {}
        for k in keys:
            v = param_space[k]
            if isinstance(v, tuple) and len(v) == 2:
                p[k] = rng.uniform(float(v[0]), float(v[1]))
            else:
                p[k] = rng.choice(list(v))
        yield p


def mutate_params(params: dict, param_space: dict, rng: random.Random, rate=0.25):
    out = dict(params)
    for k, v in param_space.items():
        if rng.random() > rate:
            continue
        if isinstance(v, tuple) and len(v) == 2:
            width = float(v[1]) - float(v[0])
            out[k] = min(float(v[1]), max(float(v[0]), float(out[k]) + rng.gauss(0, width * 0.10)))
        else:
            out[k] = rng.choice(list(v))
    return out


def genetic_candidates(param_space: dict, evaluate, *, population=24, generations=6, elite_frac=0.25, seed=42):
    rng = random.Random(seed)
    pop = list(sample_random(param_space, population, seed=seed))
    seen = set()
    for _ in range(generations):
        scored = []
        for p in pop:
            key = tuple(sorted(p.items()))
            if key in seen:
                continue
            seen.add(key)
            scored.append((evaluate(p), p))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            break
        elites = [p for _, p in scored[: max(2, int(len(scored) * elite_frac))]]
        yield from elites
        pop = elites[:]
        while len(pop) < population:
            a = rng.choice(elites)
            b = rng.choice(elites)
            child = {k: rng.choice([a.get(k), b.get(k)]) for k in param_space}
            child = mutate_params(child, param_space, rng)
            pop.append(child)


def candidate_stream(param_space: dict, optimizer="auto", max_evals=500, seed=42):
    total = 1
    for v in param_space.values():
        if isinstance(v, tuple) and len(v) == 2:
            total *= 7
        else:
            total *= len(list(v))
    if optimizer == "auto":
        optimizer = "grid" if total <= max_evals else "random"
    if optimizer == "grid":
        yield from itertools.islice(expand_grid(param_space), int(max_evals))
    elif optimizer == "random":
        yield from sample_random(param_space, int(max_evals), seed=seed)
    else:
        # fallback for optimizers that are handled outside
        yield from sample_random(param_space, int(max_evals), seed=seed)
