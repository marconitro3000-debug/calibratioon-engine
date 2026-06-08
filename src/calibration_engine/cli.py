import numpy as np
import pandas as pd

from .engine import CalibrationEngine


def main() -> None:
    x = np.linspace(0, 10, 80)
    y = 2.5 * x + 4.0
    data = pd.DataFrame({"x": x, "y": y})

    def objective(params, data):
        prediction = params["slope"] * data["x"] + params["intercept"]
        rmse = float(np.sqrt(np.mean((data["y"] - prediction) ** 2)))
        return {"score": -rmse, "rmse": rmse}

    result = CalibrationEngine().calibrate(
        objective,
        {"slope": [1.5, 2.0, 2.5, 3.0], "intercept": [2.0, 4.0, 6.0]},
        data=data,
        optimizer="grid",
    )
    print(result.report())


if __name__ == "__main__":
    main()
