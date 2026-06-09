# Model File Guide

Calibration Engine is simplest when each experiment lives in a local model file under `models/`.

## Minimal Model

```python
PARAM_SPACE = {"x": [1, 2, 3]}


def objective(params, data):
    return params["x"]
```

Run it:

```bash
calibration-engine models/my_model.py
```

The run is saved under:

```text
runs/my_model/<timestamp>/
  manifest.json
  result.json
  trials.csv
  report.md
```

`runs/` is ignored by git so local experiment artifacts are not committed by default.

## Optional Fields

```python
SEED = 42
OPTIMIZER = "random"
MAX_EVALS = 100
DIRECTION = "maximize"
METADATA = {"experiment": "my-experiment-v1"}
```

For successive halving:

```python
OPTIMIZER = "successive_halving"
MIN_RESOURCE = 1
MAX_RESOURCE = 9
REDUCTION_FACTOR = 3
RESOURCE_NAME = "_resource"
```

The objective can read:

```python
resource = params["_resource"]
```

## Data

Use `load_data()` when data construction is part of the model file:

```python
def load_data():
    return {"x": [1, 2, 3]}
```

or define static `DATA`:

```python
DATA = {"x": [1, 2, 3]}
```

## Constraints

```python
CONSTRAINTS = [
    lambda params: params["short"] < params["long"],
]
```

Invalid candidates are recorded as skipped trials.

## Security

Model files are local Python code. Only run model files you wrote or trust.
