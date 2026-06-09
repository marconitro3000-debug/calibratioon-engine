import numpy as np
import pandas as pd
import pytest

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


def test_calibration_supports_constraints_and_metadata():
    def objective(params, data):
        return params["x"] + params["y"]

    result = CalibrationEngine(seed=123).calibrate(
        objective,
        {"x": [1, 2, 3], "y": [1, 2, 3]},
        optimizer="grid",
        constraints=[lambda params: params["x"] <= params["y"]],
        metadata={"experiment": "constraint-test"},
    )

    assert result.best_params == {"x": 3, "y": 3}
    assert len(result.skipped_trials) == 3
    assert result.metadata["seed"] == 123
    assert result.metadata["experiment"] == "constraint-test"
    assert result.summary()["n_skipped"] == 3
    assert result.top_trials(1)[0].score == 6.0


def test_calibration_rejects_invalid_budget():
    with pytest.raises(ValueError):
        CalibrationEngine().calibrate(lambda params, data: 1.0, {"x": [1]}, max_evals=0)


def test_non_finite_scores_are_recorded_as_failed_trials():
    result = CalibrationEngine().calibrate(lambda params, data: np.nan, {"x": [1]}, optimizer="grid")

    assert result.failed_trials
    assert result.failed_trials[0].status == "error"


def test_all_constraints_skipped_still_returns_auditable_result():
    result = CalibrationEngine().calibrate(
        lambda params, data: params["x"],
        {"x": [1, 2]},
        constraints=[lambda params: False],
        optimizer="grid",
    )

    assert len(result.ok_trials) == 0
    assert len(result.skipped_trials) == 2
    assert result.best_params == {"x": 1}


def test_callbacks_can_observe_and_stop_trials():
    started = []
    ended = []

    result = CalibrationEngine().calibrate(
        lambda params, data: params["x"],
        {"x": [1, 2, 3]},
        optimizer="grid",
        on_trial_start=lambda params: started.append(params["x"]),
        on_trial_end=lambda trial: ended.append(trial.score),
        should_stop=lambda trials: len(trials) == 2,
    )

    assert started == [1, 2]
    assert ended == [1.0, 2.0]
    assert result.n_trials == 2
