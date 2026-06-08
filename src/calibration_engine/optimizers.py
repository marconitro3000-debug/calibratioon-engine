import itertools
import random
from collections.abc import Iterator
from typing import Any

import numpy as np


def expand_grid(param_space: dict[str, Any]) -> Iterator[dict[str, Any]]:
    keys = list(param_space)
    values = [_values_for_space(param_space[key]) for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo, strict=False))


def sample_random(param_space: dict[str, Any], n: int, seed: int = 42) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    keys = list(param_space)
    for _ in range(int(n)):
        params = {}
        for key in keys:
            value = param_space[key]
            if _is_range(value):
                params[key] = rng.uniform(float(value[0]), float(value[1]))
            else:
                params[key] = rng.choice(list(value))
        yield params


def candidate_stream(
    param_space: dict[str, Any],
    *,
    optimizer: str = "auto",
    max_evals: int = 100,
    seed: int = 42,
) -> Iterator[dict[str, Any]]:
    total = _grid_size(param_space)
    if optimizer == "auto":
        optimizer = "grid" if total <= max_evals else "random"
    if optimizer == "grid":
        yield from itertools.islice(expand_grid(param_space), int(max_evals))
    elif optimizer == "random":
        yield from sample_random(param_space, int(max_evals), seed=seed)
    else:
        raise ValueError("optimizer must be one of: auto, grid, random")


def _grid_size(param_space: dict[str, Any]) -> int:
    total = 1
    for value in param_space.values():
        total *= len(_values_for_space(value))
    return total


def _values_for_space(value: Any) -> list[Any]:
    if _is_range(value):
        return np.linspace(float(value[0]), float(value[1]), 7).tolist()
    return list(value)


def _is_range(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) == 2
