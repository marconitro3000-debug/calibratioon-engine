# OpenAlphaLab V3

OpenAlphaLab is a public Python package for reusable parameter calibration, formulaic alpha research, validation and reporting.

It is not affiliated with WorldQuant, Jane Street, Citadel or any private platform. The goal is to implement an open, reproducible version of the alpha research workflow:

```text
formula / model hypothesis
-> parameter calibration
-> neutralization
-> cost-aware backtest
-> IC / decay analysis
-> walk-forward validation
-> overfitting diagnostics
-> alpha correlation clustering
-> alpha book construction
-> audit report
```

## Core API

Generic model calibration:

```python
from alpha_lab import CalibrationEngine


def objective(params, data):
    prediction = params["slope"] * data["x"] + params["intercept"]
    rmse = ((data["y"] - prediction) ** 2).mean() ** 0.5
    return {"score": -rmse, "rmse": rmse}


result = CalibrationEngine(seed=7).calibrate(
    objective,
    param_space={
        "slope": (0.0, 5.0),
        "intercept": (-2.0, 8.0),
    },
    data=training_dataframe,
    optimizer="random",
    max_evals=200,
    direction="maximize",
)

print(result.best_params)
print(result.best_metrics)
print(result.report())
```

Alpha research:

```python
from alpha_lab import AutoAlphaResearchEngine

engine = AutoAlphaResearchEngine(
    neutralization=["market"],
    costs={"commission_bps": 1, "slippage_bps": 2},
    validation="walk_forward",
    overfitting_control=True,
)

result = engine.research(
    alpha_template="-rank(delta(close, {lookback})) * rank(volume / adv{adv_window})",
    data=data,
    param_space={
        "lookback": [1, 3, 5, 10],
        "adv_window": [10, 20, 60],
        "decay": [1, 3],
    },
    objective="robust_fitness",
)

print(result.verdict)
print(result.best_formula)
print(result.metrics)
print(result.rejection_reasons)
print(result.report())
```

`data` is a dictionary of pandas DataFrames:

```python
data = {
    "close": close,      # index=date, columns=assets
    "open": open_,
    "high": high,
    "low": low,
    "volume": volume,
}
```

## Supported Operators

- Cross-sectional: `rank`, `zscore`, `scale`, `winsorize`, `neutralize`, `group_neutralize`.
- Time-series: `delay`, `delta`, `returns`, `ts_mean`, `ts_std`, `ts_min`, `ts_max`, `ts_rank`, `ts_corr`, `ts_cov`, `decay_linear`.
- Transformations: `signed_power`.
- Convenience variables: `adv5`, `adv10`, `adv20`, `adv60` through the `advN` parser expansion.

## V3 Modules

- Safe formula parser with vectorized pandas operators.
- Cross-sectional alpha backtester with delayed execution.
- Market/group neutralization.
- Transaction costs and turnover penalty.
- IC, Rank IC, IC decay, Sharpe, drawdown, turnover, hit rate and profit factor metrics.
- Parameter calibration with grid, random, auto, successive halving and simple genetic search.
- Walk-forward validation and overfitting proxies.
- Alpha rejection framework.
- Alpha correlation filtering and alpha book construction.
- Markdown audit reports.
- CLI demo.

## Design Principle

OpenAlphaLab should reject weak alphas, not just find high in-sample Sharpe. The score penalizes turnover, cost drag, instability, out-of-sample degradation, high alpha correlation and parameter fragility.

## Run

```bash
pip install -e ".[dev]"
python examples/test_single_alpha.py
python examples/calibrate_generic_model.py
python examples/calibrate_alpha_template.py
python examples/build_alpha_book.py
python -m pytest
```

## Development Workflow

Install the project in editable mode with development tools:

```bash
python -m pip install -e ".[dev]"
```

Run the local quality gate:

```bash
python -m ruff check .
python -m black --check .
python -m mypy src/alpha_lab
python -m pytest
```

Optional pre-commit setup:

```bash
pre-commit install
pre-commit run --all-files
```

## Use From Another Project

Install from GitHub:

```bash
python -m pip install git+https://github.com/marconitro3000-debug/calibratioon-engine.git
```

Then import it normally:

```python
from alpha_lab import CalibrationEngine
```

Your other project only needs to provide:

- a parameter space;
- a function that scores one parameter set;
- optional data, usually a DataFrame, dict, CSV-loaded table or model object.

OpenAlphaLab handles trial generation, failure capture, ranking, result tables and reports.

## Security Model

This project is an offline Python library. It does not start servers, open ports, expose a public HTTP API, call remote APIs, require API keys or execute shell commands from user inputs.

See `SECURITY.md` for the full safety model.

For portfolio risk workflows, pass JSON/CSV files directly:

```python
from alpha_lab import analyze_portfolio_risk

result = analyze_portfolio_risk(
    holdings="portfolio_holdings.json",
    prices="historical_prices.csv",
    benchmark="SPY",
)

print(result.metrics)
print(result.warnings)
print(result.report())
```

Supported holdings formats:

```json
{
  "holdings": [
    {"symbol": "AAPL", "market_value": 12000},
    {"symbol": "MSFT", "market_value": 8000}
  ]
}
```

or CSV:

```csv
symbol,market_value
AAPL,12000
MSFT,8000
```

Supported price CSV format:

```csv
date,AAPL,MSFT,SPY
2024-01-01,184.2,372.1,474.0
2024-01-02,185.0,370.8,475.2
```

Run the demo:

```bash
python examples/analyze_portfolio_risk.py
```

## Project Quality Bar

A change is considered ready when:

- The public API stays importable from `alpha_lab`.
- Examples run with synthetic data.
- `python -m pytest` passes.
- CI passes lint, format, type-check and tests.
- New research logic includes tests or a documented reason why it is exploratory.
