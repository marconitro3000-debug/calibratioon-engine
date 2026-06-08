# Contributing

Calibration Engine should stay small, offline and reusable.

## Local Setup

```bash
python -m pip install -e ".[dev]"
```

## Checks

```bash
python -m ruff check .
python -m black --check .
python -m mypy src/calibration_engine
python -m pytest -q
```

## Rules

- Keep the public API importable from `calibration_engine`.
- Do not add network calls.
- Do not add servers or ports.
- Do not read credentials or `.env` files.
- Do not execute shell commands from user inputs.
- Keep optimizers deterministic when a seed is provided.
- Add tests for new optimizers, result formats or calibration behavior.
