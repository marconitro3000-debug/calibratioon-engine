# Calibration Engine: An Offline Framework for Auditable Parameter Search

## Abstract

Calibration Engine is a small Python framework for reproducible parameter calibration. It targets research workflows where a user has an objective function, a parameter space and a fixed evaluation budget. The library records every trial, skipped candidate, failed evaluation, result summary and experiment manifest so runs can be inspected after execution.

The design is intentionally offline and domain-neutral. It does not start servers, open network ports, call external APIs or require credentials.

## 1. Motivation

Research code often repeats the same calibration scaffolding:

1. define candidate parameters;
2. evaluate a model or simulation;
3. rank candidates;
4. handle failures;
5. preserve metadata;
6. export results for review.

Reimplementing this per project creates avoidable drift. Calibration Engine provides a stable calibration contract that can be reused across projects without coupling the library to any specific model domain.

## 2. Design Goals

- **Reproducibility**: every run records seed, optimizer, budget, Python version, platform and user metadata.
- **Auditability**: every trial is stored with params, score, metrics, status and error.
- **Safety**: the package is local-only and does not execute shell commands or network calls.
- **Small API surface**: users provide an objective callback and a validated `SearchSpace`.
- **Extensibility**: new optimizers should plug into the same `CalibrationResult` structure.

## 3. Methodology

The core calibration loop is:

```text
SearchSpace -> candidate stream -> objective(params, data) -> CalibrationTrial -> CalibrationResult
```

An objective can return either:

```python
float_score
```

or:

```python
{"score": float_score, "metric_name": metric_value}
```

The `score` is the scalar optimization target. Extra keys are recorded as metrics.

## 4. Search Space Representation

Parameter spaces can be discrete:

```python
{"depth": [2, 3, 4]}
```

or bounded continuous ranges:

```python
{"rate": (0.001, 0.1)}
```

For research-grade runs, `SearchSpace.from_dict(...)` creates a validated manifest. This manifest is stored in `result.metadata["search_space"]`.

## 5. Optimizers

The current version includes:

- `grid`: deterministic enumeration of the grid representation;
- `random`: seeded random sampling;
- `auto`: grid for small spaces, random for larger spaces.
- `successive_halving`: budget-aware candidate promotion over increasing resources.

This follows a conservative research workflow: simple baselines first, then more advanced optimizers behind the same result contract.

## 6. Reproducibility Contract

Each `CalibrationResult` includes:

- best params;
- best score;
- best metrics;
- all trials;
- status counts;
- score summary;
- metadata;
- search-space manifest.

Results can be saved locally:

```python
result.save_json("result.json")
result.save_csv("trials.csv")
```

and loaded again:

```python
loaded = CalibrationResult.load_json("result.json")
```

## 7. Demonstration

The notebook `notebooks/calibration_engine_research_demo.ipynb` demonstrates a synthetic regression calibration problem. It compares a grid search and a seeded random search over the same objective, then inspects top trials and summary statistics.

## 8. Limitations

- No parallel execution yet.
- No Bayesian optimizer yet.
- Continuous ranges use a compact grid approximation for grid search.
- Objective functions are user code; the library records objective failures but does not sandbox them.

## 9. Roadmap

1. Callback hooks for richer logging.
2. Experiment manifest files.
3. Parallel local execution.
4. Typed parameter distributions.
5. More examples for model calibration without adding heavy dependencies.

## 10. Security

Calibration Engine is an offline local library. It does not start servers, open ports, call remote APIs, require keys or read `.env` files.

See `SECURITY.md`.
