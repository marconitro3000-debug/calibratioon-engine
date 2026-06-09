# Research Notes

This project stays domain-neutral: it is an offline parameter calibration engine, not a model, server or data product.

The design is informed by general hyperparameter optimization and reproducibility research.

## Core References

1. James Bergstra and Yoshua Bengio, "Random Search for Hyper-Parameter Optimization", JMLR 2012.
   Source: https://jmlr.org/beta/papers/v13/bergstra12a.html

2. Jasper Snoek, Hugo Larochelle and Ryan P. Adams, "Practical Bayesian Optimization of Machine Learning Algorithms", NeurIPS 2012.
   Source: https://arxiv.org/abs/1206.2944

3. Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh and Ameet Talwalkar, "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization", JMLR 2018.
   Source: https://www.jmlr.org/beta/papers/v18/16-558.html

4. Xavier Bouthillier, Cesar Laurent and Pascal Vincent, "Unreproducible Research is Reproducible", ICML 2019.
   Source: https://proceedings.mlr.press/v97/bouthillier19a.html

## Design Implications

### Random Search Is A Strong Baseline

Bergstra and Bengio show why random search is a practical baseline when only a subset of parameters materially affects performance. For this library:

- `random` should remain a first-class optimizer, not a fallback.
- `auto` should prefer exhaustive grid only for small spaces.
- results should expose every evaluated trial so users can inspect variance and sensitivity.

### Adaptive Methods Belong Behind A Stable Result Contract

Bayesian optimization and bandit methods can be valuable, but they add assumptions and moving parts. The stable part of the package should be:

- objective callback contract;
- parameter-space representation;
- trial records;
- result summaries;
- persistence format.

Future optimizers can plug into this without changing `CalibrationResult`.

### Budget-Aware Search Matters

Hyperband frames hyperparameter optimization as a budget allocation problem. The current engine does not implement multi-fidelity scheduling yet, but it should keep the API compatible with:

- `resource_name`;
- `min_resource`;
- `max_resource`;
- `reduction_factor`;
- early stopping callbacks.

### Reproducibility Is Part Of The Product

Bouthillier, Laurent and Vincent emphasize ambiguity around reproducibility. For this package, every run should preserve:

- seed;
- optimizer;
- max evaluations;
- Python version;
- platform;
- user-provided experiment metadata;
- all trial scores, failures and skipped candidates.

That is why `CalibrationResult` records metadata, trial status, JSON persistence and CSV export.

## Near-Term Roadmap

1. Add an explicit `SearchSpace` object with validation.
2. Add deterministic stratified random sampling for mixed discrete/continuous spaces.
3. Add successive halving as the first budget-aware optimizer.
4. Add callback hooks: `on_trial_start`, `on_trial_end`, `should_stop`.
5. Add a compact experiment manifest file.
6. Add examples for scikit-learn-style models without requiring scikit-learn as a dependency.
