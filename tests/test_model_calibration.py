import numpy as np
import pandas as pd

from calibration_engine import CalibrationEngine, CalibrationResult, calibrate_model


def make_data():
    x = np.arange(20, dtype=float)
    y = 3.0 * x + 2.0
    return pd.DataFrame({"x": x, "y": y})


def linear_objective(params, data):
    prediction = params["slope"] * data["x"] + params["intercept"]
    rmse = float(np.sqrt(np.mean((data["y"] - prediction) ** 2)))
    return {"score": -rmse, "rmse": rmse}


def test_calibration_engine_finds_best_grid_params():
    result = CalibrationEngine().calibrate(
        linear_objective,
        {"slope": [1.0, 2.0, 3.0], "intercept": [0.0, 2.0, 4.0]},
        data=make_data(),
        optimizer="grid",
        direction="maximize",
    )

    assert result.best_params == {"slope": 3.0, "intercept": 2.0}
    assert result.best_metrics["rmse"] == 0.0
    assert len(result.trials) == 9
    assert not result.to_frame().empty


def test_calibrate_model_helper_supports_minimize():
    def objective(params, data):
        return (params["x"] - 4) ** 2

    result = calibrate_model(
        objective,
        {"x": [1, 2, 3, 4, 5]},
        optimizer="grid",
        direction="minimize",
    )

    assert result.best_params == {"x": 4}
    assert result.best_score == 0.0


def test_calibration_records_failed_trials():
    def objective(params, data):
        if params["x"] == 2:
            raise ValueError("bad parameter")
        return params["x"]

    result = CalibrationEngine().calibrate(
        objective,
        {"x": [1, 2, 3]},
        optimizer="grid",
    )

    assert result.best_params == {"x": 3}
    assert any(trial.status == "error" for trial in result.trials)


def test_calibration_result_saves_json_and_csv(tmp_path):
    result = CalibrationEngine().calibrate(
        linear_objective,
        {"slope": [2.0, 3.0], "intercept": [2.0]},
        data=make_data(),
        optimizer="grid",
    )

    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "trials.csv"
    result.save_json(json_path)
    result.save_csv(csv_path)
    loaded = CalibrationResult.load_json(json_path)

    assert loaded.best_params == result.best_params
    assert loaded.best_score == result.best_score
    assert loaded.n_trials == result.n_trials
    assert csv_path.read_text(encoding="utf-8").startswith("param_slope")
    assert "Failed trials" in result.report()
