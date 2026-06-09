# API Reference

## `CalibrationEngine`

```python
from calibration_engine import CalibrationEngine
```

Runs offline parameter calibration.

Main method:

```python
result = CalibrationEngine(seed=42).calibrate(
    objective,
    param_space,
    data=None,
    optimizer="auto",
    max_evals=100,
    direction="maximize",
    constraints=None,
    metadata=None,
    on_trial_start=None,
    on_trial_end=None,
    should_stop=None,
    min_resource=1,
    max_resource=None,
    reduction_factor=3,
    resource_name="_resource",
)
```

The objective receives:

```python
objective(params, data)
```

and returns either a float score or a dict with a `score` key.

## `SearchSpace`

```python
from calibration_engine import SearchSpace

space = SearchSpace.from_dict({"depth": [2, 3, 4], "rate": (0.001, 0.1)})
```

`SearchSpace` validates candidate values and bounded ranges, computes `grid_size`, and serializes a manifest into result metadata.

## Optimizers

Supported optimizers:

- `grid`
- `random`
- `auto`
- `successive_halving`

`successive_halving` evaluates many candidates with a small budget, promotes the strongest candidates, and reevaluates them with larger budgets. The current budget is injected into params using `resource_name`, defaulting to `_resource`.

## `CalibrationResult`

Returned by `CalibrationEngine.calibrate`.

Useful attributes:

- `best_params`
- `best_score`
- `best_metrics`
- `trials`
- `metadata`
- `ok_trials`
- `failed_trials`
- `skipped_trials`

Useful methods:

- `summary()`
- `top_trials(n)`
- `to_dict()`
- `to_frame()`
- `report()`
- `save_json(path)`
- `save_csv(path)`
- `CalibrationResult.load_json(path)`

## `run_model_file`

```python
from calibration_engine import run_model_file

run = run_model_file("models/example_model.py")
```

Loads a local trusted model file and writes:

- `result.json`
- `trials.csv`
- `report.md`

to `runs/<model_name>/<timestamp>/`.

Required model file fields:

- `PARAM_SPACE`
- `objective(params, data)`

Optional fields:

- `load_data()`
- `DATA`
- `SEED`
- `OPTIMIZER`
- `MAX_EVALS`
- `DIRECTION`
- `CONSTRAINTS`
- `METADATA`

Model files are local Python code. Only run model files you trust.
