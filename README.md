# Calibration Engine

Calibration Engine is a small offline Python library for reusable parameter calibration.

It is designed for projects where you already have a model, simulation or scoring function and you do not want to rewrite parameter search, trial tracking and reporting every time.

It is not a server, not a public web API, and not connected to any external service.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Basic Usage

```python
from calibration_engine import CalibrationEngine


def objective(params, data):
    prediction = params["slope"] * data["x"] + params["intercept"]
    rmse = ((data["y"] - prediction) ** 2).mean() ** 0.5
    return {"score": -rmse, "rmse": rmse}


result = CalibrationEngine(seed=7).calibrate(
    objective,
    param_space={
        "slope": (0.0, 5.0),
        "intercept": (-2.0, 8.0),
    },
    data=training_dataframe,
    optimizer="random",
    max_evals=200,
    direction="maximize",
)

print(result.best_params)
print(result.best_metrics)
print(result.report())

result.save_json("calibration_result.json")
result.save_csv("calibration_trials.csv")
```

## What It Provides

- `CalibrationEngine`: reusable calibration runner.
- `calibrate_model`: convenience function.
- `CalibrationResult`: best params, best score, metrics, all trials.
- `CalibrationTrial`: one evaluated parameter set.
- Optimizers: `grid`, `random`, `auto`.
- Result export via `to_dict()` and `to_frame()`.
- Local persistence via `save_json()`, `load_json()` and `save_csv()`.
- Markdown report via `report()`.

## Parameter Spaces

Discrete values:

```python
{"window": [5, 10, 20], "threshold": [0.1, 0.2, 0.3]}
```

Numeric ranges:

```python
{"learning_rate": (0.0001, 0.1), "depth": [2, 3, 4]}
```

Ranges are sampled for random search and expanded into a small fixed grid for grid search.

## Run Example

```bash
python examples/calibrate_generic_model.py
```

## Quality Gate

```bash
python -m ruff check .
python -m black --check .
python -m mypy src/calibration_engine
python -m pytest -q
```

## Security Model

This is an offline local library. It does not start servers, open ports, call remote APIs, require API keys, read `.env` files or execute shell commands from user inputs.

See `SECURITY.md`.
