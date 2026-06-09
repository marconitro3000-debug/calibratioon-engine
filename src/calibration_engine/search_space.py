from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    values: tuple[Any, ...] | None = None
    bounds: tuple[float, float] | None = None
    grid_points: int = 7

    @classmethod
    def from_raw(cls, name: str, raw: Any) -> ParameterSpec:
        if isinstance(raw, tuple) and len(raw) == 2:
            lower = float(raw[0])
            upper = float(raw[1])
            if lower > upper:
                raise ValueError(f"parameter '{name}' lower bound is greater than upper bound")
            return cls(name=name, bounds=(lower, upper))

        values = tuple(raw)
        if not values:
            raise ValueError(f"parameter '{name}' has no candidate values")
        return cls(name=name, values=values)

    @property
    def is_continuous(self) -> bool:
        return self.bounds is not None

    def grid_values(self) -> list[Any]:
        if self.bounds is None:
            return list(self.values or ())
        lower, upper = self.bounds
        return np.linspace(lower, upper, self.grid_points).tolist()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": list(self.values) if self.values is not None else None,
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "grid_points": self.grid_points,
        }


@dataclass(frozen=True)
class SearchSpace:
    parameters: tuple[ParameterSpec, ...]

    @classmethod
    def from_dict(cls, param_space: dict[str, Any] | SearchSpace) -> SearchSpace:
        if isinstance(param_space, SearchSpace):
            return param_space
        if not param_space:
            raise ValueError("param_space cannot be empty")
        return cls(tuple(ParameterSpec.from_raw(name, raw) for name, raw in param_space.items()))

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.parameters]

    @property
    def grid_size(self) -> int:
        total = 1
        for spec in self.parameters:
            total *= len(spec.grid_values())
        return total

    def grid_values(self) -> list[list[Any]]:
        return [spec.grid_values() for spec in self.parameters]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": [spec.to_dict() for spec in self.parameters],
            "grid_size": self.grid_size,
        }
