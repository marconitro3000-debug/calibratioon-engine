import numpy as np
import pandas as pd
from alpha_lab import AutoAlphaResearchEngine

rng = np.random.default_rng(2)
n, m = 500, 30
rets = rng.normal(0, 0.012, size=(n, m))
close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), columns=[f"A{i}" for i in range(m)])
volume = pd.DataFrame(rng.lognormal(12, 0.35, size=(n, m)), columns=close.columns)
data = {"close": close, "volume": volume}

engine = AutoAlphaResearchEngine(neutralization=["market"], costs={"commission_bps": 1, "slippage_bps": 2})
result = engine.research(
    "-rank(delta(close, {lookback})) * rank(volume / adv{adv_window})",
    data=data,
    param_space={"lookback": [1, 3, 5, 10, 20], "adv_window": [10, 20, 60], "decay": [1, 3, 5]},
    optimizer="auto",
    max_evals=300,
)
print(result.report())
