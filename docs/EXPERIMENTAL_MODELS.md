# Experimental Models

The `models/experimental/` folder contains synthetic model files inspired by recent public research in hyperparameter optimization and multi-fidelity calibration.

These are not full paper replications. They are compact offline examples that mimic the shape of research problems:

- learning curves;
- budgets/fidelities;
- noisy early-stage evaluations;
- cost-aware scoring;
- candidate promotion.

## Models

### `adaptive_fidelity_identification.py`

Inspired by:

- "Efficient Hyperparameter Optimization with Adaptive Fidelity Identification", CVPR 2024.
  Source: https://openaccess.thecvf.com/content/CVPR2024/papers/Jiang_Efficient_Hyperparameter_Optimization_with_Adaptive_Fidelity_Identification_CVPR_2024_paper.pdf

Demonstrates a synthetic objective where early resources are biased and higher resources reveal better candidates.

Run:

```bash
calibration-engine models/experimental/adaptive_fidelity_identification.py
```

### `freeze_thaw_curve.py`

Inspired by:

- "In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization", 2024.
  Source: https://arxiv.org/abs/2404.16795

Demonstrates learning-curve style evaluation with asymptote, rate and stability parameters.

Run:

```bash
calibration-engine models/experimental/freeze_thaw_curve.py
```

### `cost_aware_multifidelity.py`

Inspired by:

- "Multi-Fidelity Methods for Optimization: A Survey", 2024.
  Source: https://arxiv.org/abs/2402.09638

Demonstrates a resource-aware objective where score trades off quality against compute cost.

Run:

```bash
calibration-engine models/experimental/cost_aware_multifidelity.py
```

## Notes

All models are local Python files and use only synthetic calculations. They do not download data, call APIs or open network connections.
