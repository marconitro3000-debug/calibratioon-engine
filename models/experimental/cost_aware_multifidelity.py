import math

from calibration_engine import SearchSpace

SEED = 23
OPTIMIZER = "successive_halving"
MAX_EVALS = 40
MIN_RESOURCE = 2
MAX_RESOURCE = 18
REDUCTION_FACTOR = 3
DIRECTION = "maximize"
METADATA = {
    "experiment": "cost-aware-multifidelity-synthetic",
    "paper_inspiration": "Multi-Fidelity Methods for Optimization: A Survey, 2024",
}

PARAM_SPACE = SearchSpace.from_dict(
    {
        "solver_depth": [1, 2, 3, 4, 5],
        "exploration": [0.05, 0.1, 0.2, 0.4],
        "smoothing": [0.0, 0.05, 0.1, 0.2],
    }
)


def objective(params, data):
    resource = params["_resource"]
    solver_depth = params["solver_depth"]
    exploration = params["exploration"]
    smoothing = params["smoothing"]

    quality = 1.2 - 0.12 * (solver_depth - 3) ** 2 - 0.7 * (exploration - 0.2) ** 2
    quality -= 0.5 * (smoothing - 0.05) ** 2
    fidelity_gain = 1.0 - math.exp(-resource / (2.0 + solver_depth))
    compute_cost = 0.015 * resource * solver_depth
    score = quality * fidelity_gain - compute_cost

    return {
        "score": score,
        "resource": resource,
        "quality": quality,
        "compute_cost": compute_cost,
    }
