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
from calibration_engine import CalibrationEngine, SearchSpace


def objective(params, data):
    prediction = params["slope"] * data["x"] + params["intercept"]
    rmse = ((data["y"] - prediction) ** 2).mean() ** 0.5
    return {"score": -rmse, "rmse": rmse}


result = CalibrationEngine(seed=7).calibrate(
    objective,
    param_space=SearchSpace.from_dict({
        "slope": (0.0, 5.0),
        "intercept": (-2.0, 8.0),
    }),
    data=training_dataframe,
    optimizer="random",
    max_evals=200,
    direction="maximize",
    metadata={"experiment": "linear-demo-v1"},
)

print(result.best_params)
print(result.best_metrics)
print(result.summary())
print(result.report())

result.save_json("calibration_result.json")
result.save_csv("calibration_trials.csv")
```

## Simple Project Layout

For day-to-day work, put model definitions in `models/` and let the engine write runs to `runs/`:

```text
models/
  example_model.py
runs/
  example_model/
    20260609T120000Z/
      result.json
      trials.csv
      report.md
      manifest.json
```

Run a model file:

```bash
calibration-engine models/example_model.py
```

or from Python:

```python
from calibration_engine import run_model_file

run = run_model_file("models/example_model.py")
print(run.run_dir)
print(run.result.best_params)
```

A model file defines:

```python
PARAM_SPACE = {"x": [1, 2, 3]}

def objective(params, data):
    return params["x"]
```

Optional fields: `load_data`, `DATA`, `SEED`, `OPTIMIZER`, `MAX_EVALS`, `DIRECTION`, `CONSTRAINTS`, `METADATA`.

## What It Provides

- `CalibrationEngine`: reusable calibration runner.
- `calibrate_model`: convenience function.
- `CalibrationResult`: best params, best score, metrics, all trials.
- `CalibrationTrial`: one evaluated parameter set.
- `SearchSpace`: explicit, validated parameter-space manifest.
- Optimizers: `grid`, `random`, `auto`, `successive_halving`.
- Parameter constraints via `constraints=[...]`.
- Reproducibility metadata: seed, optimizer, Python version, platform and user metadata.
- Trial summaries via `summary()` and ranked candidates via `top_trials()`.
- Result export via `to_dict()` and `to_frame()`.
- Local persistence via `save_json()`, `load_json()` and `save_csv()`.
- Markdown report via `report()`.

## Research-Grade Usage

Use constraints when only part of the parameter space is valid:

```python
result = CalibrationEngine(seed=42).calibrate(
    objective,
    param_space={"short_window": [5, 10, 20], "long_window": [20, 50, 100]},
    constraints=[lambda params: params["short_window"] < params["long_window"]],
    metadata={"experiment": "window-search-v1"},
)
```

The result records skipped trials, failed trials, score distribution and metadata so runs can be audited later.

Budget-aware calibration:

```python
result = CalibrationEngine(seed=42).calibrate(
    objective,
    param_space={"x": [1, 2, 3, 4, 5, 6]},
    optimizer="successive_halving",
    min_resource=1,
    max_resource=9,
    reduction_factor=3,
)
```

For `successive_halving`, each objective call receives the current budget in `params["_resource"]`.

## Research Notes

See `docs/RESEARCH_NOTES.md` for the paper-backed design notes behind the optimizer choices, reproducibility metadata and future roadmap.

For a paper-style project explanation, see `docs/PAPER.md`.

For API reference, see `docs/API.md`.

For model-file workflow details, see `docs/MODEL_FILE_GUIDE.md`.

For a Jupyter walkthrough, open:

```bash
jupyter notebook notebooks/calibration_engine_research_demo.ipynb
```

Install notebook support only when needed:

```bash
python -m pip install -e ".[notebook]"
```

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

For auditable runs, use an explicit `SearchSpace`:

```python
from calibration_engine import SearchSpace

space = SearchSpace.from_dict({"depth": [2, 3, 4], "rate": (0.001, 0.1)})
```

The serialized search-space manifest is stored in `result.metadata`.

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
