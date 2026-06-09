from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from .engine import CalibrationEngine, CalibrationResult


@dataclass(frozen=True)
class ModelRun:
    model_path: Path
    run_dir: Path
    result: CalibrationResult


def run_model_file(model_path: str | Path, *, output_dir: str | Path = "runs") -> ModelRun:
    input_path = Path(model_path)
    path = input_path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix != ".py":
        raise ValueError("model file must be a Python file")

    module = _load_module(path)
    objective = _required_attr(module, "objective")
    param_space = _required_attr(module, "PARAM_SPACE")

    load_data = getattr(module, "load_data", None)
    data = load_data() if callable(load_data) else getattr(module, "DATA", None)
    seed = int(getattr(module, "SEED", 42))
    optimizer = str(getattr(module, "OPTIMIZER", "auto"))
    max_evals = int(getattr(module, "MAX_EVALS", 100))
    direction = str(getattr(module, "DIRECTION", "maximize"))
    constraints = getattr(module, "CONSTRAINTS", None)
    metadata = dict(getattr(module, "METADATA", {}))
    metadata.setdefault("model_file", str(input_path))

    result = CalibrationEngine(seed=seed).calibrate(
        objective,
        param_space=param_space,
        data=data,
        optimizer=optimizer,
        max_evals=max_evals,
        direction=direction,  # type: ignore[arg-type]
        constraints=constraints,
        metadata=metadata,
    )

    run_dir = _make_run_dir(Path(output_dir), path.stem)
    result.save_json(run_dir / "result.json")
    result.save_csv(run_dir / "trials.csv")
    (run_dir / "report.md").write_text(result.report(), encoding="utf-8")
    return ModelRun(model_path=path, run_dir=run_dir, result=result)


def _load_module(path: Path) -> ModuleType:
    module_name = f"calibration_model_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load model file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_attr(module: ModuleType, name: str) -> Any:
    if not hasattr(module, name):
        raise AttributeError(f"model file must define {name}")
    return getattr(module, name)


def _make_run_dir(output_dir: Path, model_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / model_name / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
