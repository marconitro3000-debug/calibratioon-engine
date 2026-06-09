import numpy as np
import pandas as pd

from calibration_engine import SearchSpace

SEED = 7
OPTIMIZER = "random"
MAX_EVALS = 200
DIRECTION = "maximize"
PARAM_SPACE = SearchSpace.from_dict(
    {
        "slope": (0.0, 5.0),
        "intercept": (-2.0, 8.0),
    }
)
METADATA = {"experiment": "example-linear-model"}


def load_data():
    rng = np.random.default_rng(13)
    x = np.linspace(0, 10, 120)
    y = 2.5 * x + 4.0 + rng.normal(0, 1.2, size=len(x))
    return pd.DataFrame({"x": x, "y": y})


def objective(params, data):
    prediction = params["slope"] * data["x"] + params["intercept"]
    error = data["y"] - prediction
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "score": -rmse,
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
    }
