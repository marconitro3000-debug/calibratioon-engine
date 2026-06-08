import numpy as np
import pandas as pd
import pytest

from alpha_lab import AutoAlphaResearchEngine
from alpha_lab.parser import AlphaParser


def make_data(n=80, m=6):
    rng = np.random.default_rng(11)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, m)), axis=0)),
        columns=[f"A{i}" for i in range(m)],
    )
    volume = pd.DataFrame(rng.lognormal(10, 0.2, size=(n, m)), columns=close.columns)
    return {"close": close, "volume": volume}


def test_parser_expands_adv_tokens():
    data = make_data()
    signal = AlphaParser().evaluate("rank(volume / adv20)", data)
    assert signal.shape == data["close"].shape
    assert signal.index.equals(data["close"].index)
    assert signal.columns.equals(data["close"].columns)


def test_parser_does_not_expose_builtins():
    data = make_data()
    with pytest.raises(NameError):
        AlphaParser().evaluate("__import__('os').system('echo unsafe')", data)


def test_engine_metrics_contract():
    data = make_data()
    result = AutoAlphaResearchEngine(neutralization=["market"], costs={"commission_bps": 1}).test_alpha(
        "-rank(delta(close, 3)) * rank(volume / adv20)",
        data,
    )
    expected = {
        "net_sharpe",
        "turnover",
        "cost_drag",
        "IC",
        "rank_IC",
        "ICIR",
        "alpha_decay",
        "fitness",
    }
    assert expected.issubset(result.metrics)
    assert result.positions.shape == data["close"].shape
