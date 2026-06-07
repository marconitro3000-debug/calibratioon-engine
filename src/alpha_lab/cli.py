import argparse
import numpy as np
import pandas as pd
from .engine import AutoAlphaResearchEngine


def make_demo_data(n=500, m=20, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, size=(n, m))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), columns=[f"A{i}" for i in range(m)])
    volume = pd.DataFrame(rng.lognormal(12, 0.3, size=(n, m)), columns=close.columns)
    return {"close": close, "volume": volume}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    data = make_demo_data()
    engine = AutoAlphaResearchEngine(neutralization=["market"], costs={"commission_bps": 1, "slippage_bps": 2})
    result = engine.research(
        "-rank(delta(close, {lookback})) * rank(volume / adv{adv_window})",
        data=data,
        param_space={"lookback": [1, 3, 5, 10], "adv_window": [10, 20, 60], "decay": [1, 3]},
        max_evals=100,
    )
    print(result.report())

if __name__ == "__main__":
    main()
