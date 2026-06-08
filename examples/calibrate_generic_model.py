from pathlib import Path

import numpy as np
import pandas as pd

from calibration_engine import CalibrationEngine


def make_training_csv(path: Path) -> None:
    rng = np.random.default_rng(13)
    x = np.linspace(0, 10, 120)
    y = 2.5 * x + 4.0 + rng.normal(0, 1.2, size=len(x))
    pd.DataFrame({"x": x, "y": y}).to_csv(path, index=False)


def objective(params, data):
    prediction = params["slope"] * data["x"] + params["intercept"]
    error = data["y"] - prediction
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "score": -rmse,
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
    }


def main() -> None:
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "model_training_data.csv"
    make_training_csv(csv_path)

    data = pd.read_csv(csv_path)
    result = CalibrationEngine(seed=7).calibrate(
        objective,
        param_space={
            "slope": (0.0, 5.0),
            "intercept": (-2.0, 8.0),
        },
        data=data,
        optimizer="random",
        max_evals=200,
        direction="maximize",
    )
    print(result.report())


if __name__ == "__main__":
    main()
