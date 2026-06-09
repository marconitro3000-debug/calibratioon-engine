from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from typing import Any

import numpy as np

from .optimizers import candidate_stream, expand_grid, sample_random
from .result import CalibrationResult, CalibrationTrial, ObjectiveDirection
from .search_space import SearchSpace

ObjectiveFn = Callable[[dict[str, Any], Any], float | dict[str, Any]]
ConstraintFn = Callable[[dict[str, Any]], bool]
TrialStartCallback = Callable[[dict[str, Any]], None]
TrialEndCallback = Callable[[CalibrationTrial], None]
StopCallback = Callable[[list[CalibrationTrial]], bool]


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
        min_resource: int = 1,
        max_resource: int | None = None,
        reduction_factor: int = 3,
        resource_name: str = "_resource",
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
        max_resource = max_resource or max(1, min_resource)

        if optimizer == "successive_halving":
            trials = self._run_successive_halving(
                objective=objective,
                search_space=search_space,
                data=data,
                max_evals=max_evals,
                direction=direction,
                fail_score=float(fail_score),
                constraints=constraints,
                on_trial_start=on_trial_start,
                on_trial_end=on_trial_end,
                should_stop=should_stop,
                min_resource=min_resource,
                max_resource=max_resource,
                reduction_factor=reduction_factor,
                resource_name=resource_name,
            )
        else:
            for params in candidate_stream(search_space, optimizer=optimizer, max_evals=max_evals, seed=self.seed):
                trial = _evaluate_trial(
                    objective=objective,
                    params=dict(params),
                    data=data,
                    fail_score=float(fail_score),
                    constraints=constraints,
                    on_trial_start=on_trial_start,
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
            best_params={key: value for key, value in best.params.items() if key != resource_name},
            best_score=best.score,
            best_metrics=best.metrics,
            trials=trials,
            direction=direction,
            optimizer=optimizer,
            metadata={
                "seed": self.seed,
                "optimizer": optimizer,
                "max_evals": max_evals,
                "min_resource": min_resource if optimizer == "successive_halving" else None,
                "max_resource": max_resource if optimizer == "successive_halving" else None,
                "reduction_factor": reduction_factor if optimizer == "successive_halving" else None,
                "resource_name": resource_name if optimizer == "successive_halving" else None,
                "search_space": search_space.to_dict(),
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                **(metadata or {}),
            },
        )

    def _run_successive_halving(
        self,
        *,
        objective: ObjectiveFn,
        search_space: SearchSpace,
        data: Any,
        max_evals: int,
        direction: ObjectiveDirection,
        fail_score: float,
        constraints: list[ConstraintFn],
        on_trial_start: TrialStartCallback | None,
        on_trial_end: TrialEndCallback | None,
        should_stop: StopCallback | None,
        min_resource: int,
        max_resource: int,
        reduction_factor: int,
        resource_name: str,
    ) -> list[CalibrationTrial]:
        if min_resource <= 0:
            raise ValueError("min_resource must be positive")
        if max_resource < min_resource:
            raise ValueError("max_resource must be >= min_resource")
        if reduction_factor < 2:
            raise ValueError("reduction_factor must be >= 2")

        if search_space.grid_size <= max_evals:
            candidates = list(expand_grid(search_space))
        else:
            candidates = list(sample_random(search_space, max_evals, seed=self.seed))

        trials: list[CalibrationTrial] = []
        resource = min_resource
        active = candidates
        while active:
            rung_trials = []
            for base_params in active:
                params = dict(base_params)
                params[resource_name] = resource
                trial = _evaluate_trial(
                    objective=objective,
                    params=params,
                    data=data,
                    fail_score=fail_score,
                    constraints=constraints,
                    on_trial_start=on_trial_start,
                )
                trials.append(trial)
                rung_trials.append(trial)
                if on_trial_end is not None:
                    on_trial_end(trial)
                if should_stop is not None and should_stop(trials):
                    return trials

            ok_trials = [trial for trial in rung_trials if trial.status == "ok"]
            if resource >= max_resource or len(ok_trials) <= 1:
                break

            keep = max(1, len(ok_trials) // reduction_factor)
            reverse = direction == "maximize"
            promoted = sorted(ok_trials, key=lambda trial: trial.score, reverse=reverse)[:keep]
            active = [{key: value for key, value in trial.params.items() if key != resource_name} for trial in promoted]
            resource = min(max_resource, resource * reduction_factor)
        return trials


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
    min_resource: int = 1,
    max_resource: int | None = None,
    reduction_factor: int = 3,
    resource_name: str = "_resource",
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
        min_resource=min_resource,
        max_resource=max_resource,
        reduction_factor=reduction_factor,
        resource_name=resource_name,
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


def _evaluate_trial(
    *,
    objective: ObjectiveFn,
    params: dict[str, Any],
    data: Any,
    fail_score: float,
    constraints: list[ConstraintFn],
    on_trial_start: TrialStartCallback | None,
) -> CalibrationTrial:
    if not all(constraint(params) for constraint in constraints):
        return CalibrationTrial(
            params=params,
            score=float(fail_score),
            status="skipped",
            error="parameter constraints not satisfied",
        )
    try:
        if on_trial_start is not None:
            on_trial_start(params)
        raw = objective(params, data)
        score, metrics = _parse_objective_output(raw)
        return CalibrationTrial(params=params, score=score, metrics=metrics)
    except Exception as exc:
        return CalibrationTrial(
            params=params,
            score=float(fail_score),
            status="error",
            error=str(exc),
        )
