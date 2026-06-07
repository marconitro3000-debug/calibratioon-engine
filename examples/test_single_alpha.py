import numpy as np
import pandas as pd
from alpha_lab import AutoAlphaResearchEngine

rng = np.random.default_rng(1)
n, m = 400, 25
returns = rng.normal(0, 0.01, size=(n, m))
close = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), columns=[f"S{i}" for i in range(m)])
volume = pd.DataFrame(rng.lognormal(12, 0.4, size=(n, m)), columns=close.columns)
data = {"close": close, "volume": volume}

engine = AutoAlphaResearchEngine(neutralization=["market"], costs={"commission_bps": 1, "slippage_bps": 1})
result = engine.test_alpha("-rank(delta(close, 5)) * rank(volume / adv20)", data)
print(result.report())
