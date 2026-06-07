# OpenAlphaLab V3

OpenAlphaLab is a public Python package for formulaic alpha research, calibration, validation, rejection and alpha book construction.

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
python examples/calibrate_alpha_template.py
python examples/build_alpha_book.py
python -m pytest
```
