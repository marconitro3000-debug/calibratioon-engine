from pathlib import Path

from calibration_engine import run_model_file
from calibration_engine.cli import main


def test_run_model_file_writes_artifacts(tmp_path):
    model_path = tmp_path / "simple_model.py"
    model_path.write_text(
        """
PARAM_SPACE = {"x": [1, 2, 3]}
OPTIMIZER = "grid"
MAX_EVALS = 3
METADATA = {"experiment": "runner-test"}

def objective(params, data):
    return params["x"]
""".strip(),
        encoding="utf-8",
    )

    run = run_model_file(model_path, output_dir=tmp_path / "runs")

    assert run.result.best_params == {"x": 3}
    assert (run.run_dir / "result.json").exists()
    assert (run.run_dir / "trials.csv").exists()
    assert (run.run_dir / "report.md").exists()
    assert (run.run_dir / "manifest.json").exists()
    assert Path(run.run_dir).name
    assert "runner-test" in (run.run_dir / "manifest.json").read_text(encoding="utf-8")


def test_cli_runs_model_file(tmp_path, capsys):
    model_path = tmp_path / "cli_model.py"
    model_path.write_text(
        """
PARAM_SPACE = {"x": [1, 2]}
OPTIMIZER = "grid"

def objective(params, data):
    return params["x"]
""".strip(),
        encoding="utf-8",
    )

    main([str(model_path), "--output-dir", str(tmp_path / "runs")])
    output = capsys.readouterr().out

    assert "Run saved to:" in output
    assert "Best params" in output
