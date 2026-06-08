import numpy as np
import pandas as pd

from alpha_lab import AutoAlphaResearchEngine


def make_data():
    rng = np.random.default_rng(4)
    n, m = 180, 12
    rets = rng.normal(0, 0.01, size=(n, m))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), columns=[f"A{i}" for i in range(m)])
    volume = pd.DataFrame(rng.lognormal(10, 0.2, size=(n, m)), columns=close.columns)
    return {"close": close, "volume": volume}


def test_single_alpha_runs():
    data = make_data()
    engine = AutoAlphaResearchEngine(neutralization=["market"], costs={"commission_bps": 1})
    result = engine.test_alpha("-rank(delta(close, 3)) * rank(volume / adv20)", data)
    assert "net_sharpe" in result.metrics
    assert result.positions.shape == data["close"].shape


def test_calibration_runs():
    data = make_data()
    engine = AutoAlphaResearchEngine(neutralization=["market"], costs={"commission_bps": 1})
    result = engine.research(
        "-rank(delta(close, {lookback})) * rank(volume / adv{adv_window})",
        data=data,
        param_space={"lookback": [1, 3], "adv_window": [10, 20], "decay": [1, 3]},
        max_evals=20,
    )
    assert result.best_params
    assert result.best_formula
    assert "parameter_stability" in result.metrics
