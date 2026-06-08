# Contributing

OpenAlphaLab is intended to be a usable research engine, not only a notebook demo. Contributions should keep the package importable, tested and explainable.

## Local Setup

```bash
python -m pip install -e ".[dev]"
```

## Checks

Run these before opening a PR or pushing a meaningful change:

```bash
python -m ruff check .
python -m black --check .
python -m mypy src/alpha_lab
python -m pytest
```

If pre-commit is installed:

```bash
pre-commit install
pre-commit run --all-files
```

## Engineering Rules

- Keep the top-level API stable: `from alpha_lab import AutoAlphaResearchEngine`.
- Prefer vectorized pandas operations for operators and backtests.
- Avoid look-ahead bias: trading positions must be delayed before applying returns.
- Costs must be based on turnover, not on return magnitude.
- Calibration should optimize robust fitness, not only in-sample Sharpe.
- Add tests for parser behavior, operator shape/alignment, backtest timing and alpha book selection.
- Document research assumptions in `research/` when adding new methodology.

## Release Checklist

1. Bump the version in `pyproject.toml`.
2. Run all checks.
3. Confirm examples still run.
4. Update `README.md` if the public API changed.
5. Tag the release after CI passes.
