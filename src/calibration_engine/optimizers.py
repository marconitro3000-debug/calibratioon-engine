import itertools
import random
from collections.abc import Iterator
from typing import Any

from .search_space import SearchSpace


def expand_grid(param_space: dict[str, Any] | SearchSpace) -> Iterator[dict[str, Any]]:
    space = SearchSpace.from_dict(param_space)
    keys = space.names
    values = space.grid_values()
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo, strict=False))


def sample_random(param_space: dict[str, Any] | SearchSpace, n: int, seed: int = 42) -> Iterator[dict[str, Any]]:
    space = SearchSpace.from_dict(param_space)
    rng = random.Random(seed)
    for _ in range(int(n)):
        params = {}
        for spec in space.parameters:
            if spec.bounds is not None:
                params[spec.name] = rng.uniform(spec.bounds[0], spec.bounds[1])
            else:
                params[spec.name] = rng.choice(list(spec.values or ()))
        yield params


def candidate_stream(
    param_space: dict[str, Any] | SearchSpace,
    *,
    optimizer: str = "auto",
    max_evals: int = 100,
    seed: int = 42,
) -> Iterator[dict[str, Any]]:
    space = SearchSpace.from_dict(param_space)
    total = space.grid_size
    if optimizer == "auto":
        optimizer = "grid" if total <= max_evals else "random"
    if optimizer == "grid":
        yield from itertools.islice(expand_grid(space), int(max_evals))
    elif optimizer == "random":
        yield from sample_random(space, int(max_evals), seed=seed)
    else:
        raise ValueError("optimizer must be one of: auto, grid, random")
