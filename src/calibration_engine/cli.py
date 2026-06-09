import argparse
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .engine import CalibrationEngine
from .model_runner import run_model_file


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local offline parameter calibration.")
    parser.add_argument("model", nargs="?", help="Path to a local model file, for example models/example_model.py")
    parser.add_argument("--output-dir", default="runs", help="Directory where run artifacts are written")
    args = parser.parse_args(argv)

    if args.model:
        run = run_model_file(args.model, output_dir=args.output_dir)
        print(f"Run saved to: {run.run_dir}")
        print(run.result.report(top_n=3))
        return

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
