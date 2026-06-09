from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

ObjectiveDirection = Literal["maximize", "minimize"]


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
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def failed_trials(self) -> list[CalibrationTrial]:
        return [trial for trial in self.trials if trial.status not in {"ok", "skipped"}]

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
        return {
            "n_trials": self.n_trials,
            "n_ok": len(self.ok_trials),
            "n_failed": len(self.failed_trials),
            "n_skipped": len(self.skipped_trials),
            "status_counts": dict(Counter(trial.status for trial in self.trials)),
            "best_score": self.best_score,
            "score_mean": float(scores.mean()) if not scores.empty else None,
            "score_std": float(scores.std()) if len(scores) > 1 else None,
            "score_min": float(scores.min()) if not scores.empty else None,
            "score_max": float(scores.max()) if not scores.empty else None,
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    **{f"param_{key}": value for key, value in trial.params.items()},
                    **{f"metric_{key}": value for key, value in trial.metrics.items()},
                    "score": trial.score,
                    "status": trial.status,
                    "error": trial.error,
                }
                for trial in self.trials
            ]
        )

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
        lines.extend(_format_mapping(self.best_metrics) if self.best_metrics else ["- none"])
        lines += ["", "## Summary", *_format_mapping(self.summary())]
        if self.metadata:
            lines += ["", "## Metadata", *_format_mapping(self.metadata)]
        lines += ["", f"## Top {top_n} Trials"]
        lines.extend(
            f"- score={trial.score:.6f}, params={trial.params}, status={trial.status}"
            for trial in self.top_trials(top_n)
        )
        return "\n".join(lines)


def _format_mapping(mapping: dict[str, Any]) -> list[str]:
    return [f"- {key}: {value}" for key, value in mapping.items()]
