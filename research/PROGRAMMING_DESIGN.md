# Programming Design

## API principles

The user should be able to test a single alpha:

```python
result = engine.test_alpha("-rank(delta(close, 5))", data)
```

Or calibrate a template:

```python
result = engine.research(
    "-rank(delta(close, {lookback})) * rank(volume / adv{adv_window})",
    data=data,
    param_space={"lookback": [1, 3, 5], "adv_window": [20, 60]},
)
```

## Internal flow

```text
formula/template
→ parser
→ alpha signal matrix
→ neutralization
→ portfolio construction
→ cost model
→ metrics
→ walk-forward validation
→ overfitting diagnostics
→ result report
```

## V3 architecture

- `operators.py`: vectorized alpha operators.
- `parser.py`: formula evaluation.
- `backtest.py`: positions and PnL.
- `metrics.py`: IC, Sharpe, drawdown, turnover.
- `calibration.py`: parameter calibration.
- `clustering.py`: alpha book selection.
- `engine.py`: user-facing orchestration.
