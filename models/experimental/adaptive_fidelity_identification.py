import math

from calibration_engine import SearchSpace

SEED = 31
OPTIMIZER = "successive_halving"
MAX_EVALS = 36
MIN_RESOURCE = 1
MAX_RESOURCE = 27
REDUCTION_FACTOR = 3
DIRECTION = "maximize"
METADATA = {
    "experiment": "adaptive-fidelity-identification-synthetic",
    "paper_inspiration": "Efficient Hyperparameter Optimization with Adaptive Fidelity Identification, CVPR 2024",
}

PARAM_SPACE = SearchSpace.from_dict(
    {
        "capacity": [0.4, 0.7, 1.0, 1.3, 1.6, 2.0],
        "regularization": [0.001, 0.01, 0.05, 0.1, 0.2, 0.4],
    }
)


def objective(params, data):
    resource = params["_resource"]
    capacity = params["capacity"]
    regularization = params["regularization"]

    optimum = -((capacity - 1.3) ** 2) - 0.8 * ((math.log10(regularization) + 1.0) ** 2)
    learning_progress = 1.0 - math.exp(-resource / (3.0 + 4.0 * capacity))
    early_bias = 0.04 * math.sin(resource * capacity)
    score = optimum * learning_progress + early_bias

    return {
        "score": score,
        "resource": resource,
        "learning_progress": learning_progress,
        "capacity_penalty": (capacity - 1.3) ** 2,
    }
