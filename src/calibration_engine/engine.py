from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .optimizers import candidate_stream

ObjectiveDirection = Literal["maximize", "minimize"]
ObjectiveFn = Callable[[dict[str, Any], Any], float | dict[str, Any]]


@dataclass
class CalibrationTrial:
    params: dict[str, Any]
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None


@dataclass
class CalibrationResult:
    best_params: dict[str, Any]
    best_score: float
    best_metrics: dict[str, Any]
    trials: list[CalibrationTrial]
    direction: ObjectiveDirection
    optimizer: str

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def failed_trials(self) -> list[CalibrationTrial]:
        return [trial for trial in self.trials if trial.status != "ok"]

    @property
    def ok_trials(self) -> list[CalibrationTrial]:
        return [trial for trial in self.trials if trial.status == "ok"]

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
            "",
            "## Best Metrics",
        ]
        if self.best_metrics:
            for key, value in self.best_metrics.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- none")

        lines += ["", f"## Top {top_n} Trials"]
        ranked = sorted(self.trials, key=lambda trial: trial.score, reverse=self.direction == "maximize")
        for trial in ranked[:top_n]:
            lines.append(f"- score={trial.score:.6f}, params={trial.params}, status={trial.status}")
        return "\n".join(lines)


class CalibrationEngine:
    """Offline parameter calibration engine for reusable model tuning."""

    def __init__(self, *, seed: int = 42):
        self.seed = seed

    def calibrate(
        self,
        objective: ObjectiveFn,
        param_space: dict[str, Any],
        *,
        data: Any = None,
        optimizer: str = "auto",
        max_evals: int = 100,
        direction: ObjectiveDirection = "maximize",
        fail_score: float | None = None,
    ) -> CalibrationResult:
        if direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be 'maximize' or 'minimize'")
        if not param_space:
            raise ValueError("param_space cannot be empty")

        default_fail_score = -np.inf if direction == "maximize" else np.inf
        fail_score = default_fail_score if fail_score is None else fail_score
        trials: list[CalibrationTrial] = []

        for params in candidate_stream(param_space, optimizer=optimizer, max_evals=max_evals, seed=self.seed):
            try:
                raw = objective(dict(params), data)
                score, metrics = _parse_objective_output(raw)
                trials.append(CalibrationTrial(params=dict(params), score=score, metrics=metrics))
            except Exception as exc:
                trials.append(
                    CalibrationTrial(
                        params=dict(params),
                        score=float(fail_score),
                        status="error",
                        error=str(exc),
                    )
                )

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
) -> CalibrationResult:
    return CalibrationEngine(seed=seed).calibrate(
        objective,
        param_space,
        data=data,
        optimizer=optimizer,
        max_evals=max_evals,
        direction=direction,
    )


def _parse_objective_output(raw: float | dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if isinstance(raw, dict):
        if "score" not in raw:
            raise ValueError("objective dict output must include a 'score' key")
        metrics = {key: value for key, value in raw.items() if key != "score"}
        return float(raw["score"]), metrics
    return float(raw), {}
