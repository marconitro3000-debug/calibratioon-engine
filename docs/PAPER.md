# Calibration Engine: Offline Parameter Search With Auditable Runs

## Abstract

Calibration Engine is a compact Python framework for reproducible parameter calibration. A user provides an objective function, a search space and an evaluation budget. The library returns ranked trials, summary statistics, metadata and local artifacts.

The package is offline and domain-neutral. It does not start servers, open ports, call external APIs or require credentials.

## Problem

Calibration code is often duplicated across projects. Common requirements are stable:

1. define a parameter space;
2. evaluate candidates;
3. handle failures and invalid candidates;
4. preserve a complete trial log;
5. save outputs for review.

This project factors those mechanics into a reusable library.

## Core Loop

```text
SearchSpace -> candidate generator -> objective(params, data) -> CalibrationTrial -> CalibrationResult
```

An objective returns either a scalar score or a mapping with a required `score` key:

```python
{"score": score, "metric": value}
```

## Search Space

`SearchSpace` validates discrete values and bounded numeric ranges. It also creates a manifest stored in result metadata.

```python
SearchSpace.from_dict({"depth": [2, 3, 4], "rate": (0.001, 0.1)})
```

## Optimizers

Current optimizers:

- `grid`: deterministic enumeration;
- `random`: seeded random sampling;
- `auto`: grid for small spaces, random otherwise;
- `successive_halving`: promotes candidates across increasing resource budgets.

## Run Artifacts

The local model workflow writes:

```text
runs/<model>/<timestamp>/
  manifest.json
  result.json
  trials.csv
  report.md
```

`manifest.json` records configuration and summary metadata. `trials.csv` preserves the trial table. `result.json` is a reloadable result object.

## Limitations

- No parallel execution yet.
- No Bayesian optimizer yet.
- Objective functions are trusted local Python code.
- Continuous ranges use a compact grid approximation for grid search.

## Roadmap

1. Parallel local execution.
2. Typed parameter distributions.
3. Stratified random sampling.
4. Richer callback context.
5. Additional dependency-light examples.
