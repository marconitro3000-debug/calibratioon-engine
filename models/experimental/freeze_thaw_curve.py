import math

from calibration_engine import SearchSpace

SEED = 17
OPTIMIZER = "successive_halving"
MAX_EVALS = 32
MIN_RESOURCE = 1
MAX_RESOURCE = 16
REDUCTION_FACTOR = 2
DIRECTION = "maximize"
METADATA = {
    "experiment": "freeze-thaw-curve-synthetic",
    "paper_inspiration": "In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization, 2024",
}

PARAM_SPACE = SearchSpace.from_dict(
    {
        "asymptote": [0.70, 0.78, 0.84, 0.90, 0.94],
        "rate": [0.08, 0.16, 0.32, 0.48],
        "stability": [0.02, 0.05, 0.10],
    }
)


def objective(params, data):
    resource = params["_resource"]
    asymptote = params["asymptote"]
    rate = params["rate"]
    stability = params["stability"]

    curve_value = asymptote * (1.0 - math.exp(-rate * resource))
    instability_penalty = stability * math.log1p(resource)
    overfast_penalty = 0.08 * max(0.0, rate - 0.32)
    score = curve_value - instability_penalty - overfast_penalty

    return {
        "score": score,
        "resource": resource,
        "curve_value": curve_value,
        "instability_penalty": instability_penalty,
    }
