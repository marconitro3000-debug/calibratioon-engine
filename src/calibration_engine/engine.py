from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .optimizers import candidate_stream
from .search_space import SearchSpace

ObjectiveDirection = Literal["maximize", "minimize"]
ObjectiveFn = Callable[[dict[str, Any], Any], float | dict[str, Any]]
ConstraintFn = Callable[[dict[str, Any]], bool]
TrialStartCallback = Callable[[dict[str, Any]], None]


@dataclass
class CalibrationTrial:
    params: dict[str, Any]
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None


TrialEndCallback = Callable[[CalibrationTrial], None]
StopCallback = Callable[[list[CalibrationTrial]], bool]


@dataclass
class CalibrationResult:
    best_params: dict[str, Any]
    best_score: float
    best_metrics: dict[str, Any]
    trials: list[CalibrationTrial]
    direction: ObjectiveDirection
    optimizer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def failed_trials(self) -> list[CalibrationTrial]:
        return [trial for trial in self.trials if trial.status != "ok"]

    @property
    def ok_trials(self) -> list[CalibrationTrial]:
        return [trial for trial in self.trials if trial.status == "ok"]

    @property
    def skipped_trials(self) -> list[CalibrationTrial]:
        return [trial for trial in self.trials if trial.status == "skipped"]

    def top_trials(self, n: int = 5) -> list[CalibrationTrial]:
        return sorted(self.ok_trials, key=lambda trial: trial.score, reverse=self.direction == "maximize")[:n]

    def summary(self) -> dict[str, Any]:
        scores = pd.Series([trial.score for trial in self.ok_trials], dtype=float)
        status_counts = dict(Counter(trial.status for trial in self.trials))
        return {
            "n_trials": self.n_trials,
            "n_ok": len(self.ok_trials),
            "n_failed": len(self.failed_trials),
            "n_skipped": len(self.skipped_trials),
            "status_counts": status_counts,
            "best_score": self.best_score,
            "score_mean": float(scores.mean()) if not scores.empty else None,
            "score_std": float(scores.std()) if len(scores) > 1 else None,
            "score_min": float(scores.min()) if not scores.empty else None,
            "score_max": float(scores.max()) if not scores.empty else None,
        }

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for trial in self.trials:
            rows.append(
                {
                    **{f"param_{key}": value for key, value in trial.params.items()},
                    **{f"metric_{key}": value for key, value in trial.metrics.items()},
                    "score": trial.score,
                    "status": trial.status,
                    "error": trial.error,
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "direction": self.direction,
            "optimizer": self.optimizer,
            "metadata": self.metadata,
            "trials": [
                {
                    "params": trial.params,
                    "score": trial.score,
                    "metrics": trial.metrics,
                    "status": trial.status,
                    "error": trial.error,
                }
                for trial in self.trials
            ],
        }

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def save_csv(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(output_path, index=False)

    @classmethod
    def load_json(cls, path: str | Path) -> CalibrationResult:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            best_params=payload["best_params"],
            best_score=float(payload["best_score"]),
            best_metrics=payload.get("best_metrics", {}),
            trials=[
                CalibrationTrial(
                    params=trial["params"],
                    score=float(trial["score"]),
                    metrics=trial.get("metrics", {}),
                    status=trial.get("status", "ok"),
                    error=trial.get("error"),
                )
                for trial in payload.get("trials", [])
            ],
            direction=payload["direction"],
            optimizer=payload["optimizer"],
            metadata=payload.get("metadata", {}),
        )

    def report(self, top_n: int = 5) -> str:
        lines = [
            "# Calibration Report",
            "",
            f"Optimizer: `{self.optimizer}`",
            f"Direction: `{self.direction}`",
            f"Best score: `{self.best_score:.6f}`",
            f"Best params: `{self.best_params}`",
            f"Trials: `{self.n_trials}`",
            f"Failed trials: `{len(self.failed_trials)}`",
            f"Skipped trials: `{len(self.skipped_trials)}`",
            "",
            "## Best Metrics",
        ]
        if self.best_metrics:
            for key, value in self.best_metrics.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- none")

        lines += ["", "## Summary"]
        for key, value in self.summary().items():
            lines.append(f"- {key}: {value}")

        if self.metadata:
            lines += ["", "## Metadata"]
            for key, value in self.metadata.items():
                lines.append(f"- {key}: {value}")

        lines += ["", f"## Top {top_n} Trials"]
        for trial in self.top_trials(top_n):
            lines.append(f"- score={trial.score:.6f}, params={trial.params}, status={trial.status}")
        return "\n".join(lines)


class CalibrationEngine:
    """Offline parameter calibration engine for reusable model tuning."""

    def __init__(self, *, seed: int = 42):
        self.seed = seed

    def calibrate(
        self,
        objective: ObjectiveFn,
        param_space: dict[str, Any] | SearchSpace,
        *,
        data: Any = None,
        optimizer: str = "auto",
        max_evals: int = 100,
        direction: ObjectiveDirection = "maximize",
        fail_score: float | None = None,
        constraints: list[ConstraintFn] | None = None,
        metadata: dict[str, Any] | None = None,
        on_trial_start: TrialStartCallback | None = None,
        on_trial_end: TrialEndCallback | None = None,
        should_stop: StopCallback | None = None,
    ) -> CalibrationResult:
        if direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be 'maximize' or 'minimize'")
        if max_evals <= 0:
            raise ValueError("max_evals must be positive")
        search_space = SearchSpace.from_dict(param_space)

        default_fail_score = -np.inf if direction == "maximize" else np.inf
        fail_score = default_fail_score if fail_score is None else fail_score
        trials: list[CalibrationTrial] = []
        constraints = constraints or []

        for params in candidate_stream(search_space, optimizer=optimizer, max_evals=max_evals, seed=self.seed):
            params = dict(params)
            if not all(constraint(params) for constraint in constraints):
                trial = CalibrationTrial(
                    params=params,
                    score=float(fail_score),
                    status="skipped",
                    error="parameter constraints not satisfied",
                )
                trials.append(trial)
                if on_trial_end is not None:
                    on_trial_end(trial)
                if should_stop is not None and should_stop(trials):
                    break
                continue
            try:
                if on_trial_start is not None:
                    on_trial_start(params)
                raw = objective(params, data)
                score, metrics = _parse_objective_output(raw)
                trial = CalibrationTrial(params=params, score=score, metrics=metrics)
            except Exception as exc:
                trial = CalibrationTrial(
                    params=params,
                    score=float(fail_score),
                    status="error",
                    error=str(exc),
                )
            trials.append(trial)
            if on_trial_end is not None:
                on_trial_end(trial)
            if should_stop is not None and should_stop(trials):
                break

        if not trials:
            raise ValueError("no calibration trials evaluated")

        ok_trials = [trial for trial in trials if trial.status == "ok"]
        candidates = ok_trials or trials
        reverse = direction == "maximize"
        best = sorted(candidates, key=lambda trial: trial.score, reverse=reverse)[0]

        return CalibrationResult(
            best_params=best.params,
            best_score=best.score,
            best_metrics=best.metrics,
            trials=trials,
            direction=direction,
            optimizer=optimizer,
            metadata={
                "seed": self.seed,
                "optimizer": optimizer,
                "max_evals": max_evals,
                "search_space": search_space.to_dict(),
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                **(metadata or {}),
            },
        )


def calibrate_model(
    objective: ObjectiveFn,
    param_space: dict[str, Any],
    *,
    data: Any = None,
    optimizer: str = "auto",
    max_evals: int = 100,
    direction: ObjectiveDirection = "maximize",
    seed: int = 42,
    constraints: list[ConstraintFn] | None = None,
    metadata: dict[str, Any] | None = None,
    on_trial_start: TrialStartCallback | None = None,
    on_trial_end: TrialEndCallback | None = None,
    should_stop: StopCallback | None = None,
) -> CalibrationResult:
    return CalibrationEngine(seed=seed).calibrate(
        objective,
        param_space,
        data=data,
        optimizer=optimizer,
        max_evals=max_evals,
        direction=direction,
        constraints=constraints,
        metadata=metadata,
        on_trial_start=on_trial_start,
        on_trial_end=on_trial_end,
        should_stop=should_stop,
    )


def _parse_objective_output(raw: float | dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if isinstance(raw, dict):
        if "score" not in raw:
            raise ValueError("objective dict output must include a 'score' key")
        metrics = {key: value for key, value in raw.items() if key != "score"}
        score = float(raw["score"])
    else:
        score = float(raw)
        metrics = {}
    if not np.isfinite(score):
        raise ValueError("objective score must be finite")
    return score, metrics
