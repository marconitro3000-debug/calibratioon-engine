import numpy as np
import pandas as pd
from alpha_lab import AutoAlphaResearchEngine

rng = np.random.default_rng(3)
n, m = 500, 25
rets = rng.normal(0, 0.01, size=(n, m))
close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), columns=[f"S{i}" for i in range(m)])
volume = pd.DataFrame(rng.lognormal(12, 0.3, size=(n, m)), columns=close.columns)
data = {"close": close, "volume": volume}

engine = AutoAlphaResearchEngine(neutralization=["market"], costs={"commission_bps": 1, "slippage_bps": 2})
templates = [
    "-rank(delta(close, {w}))",
    "rank(ts_corr(close, volume, {w}))",
    "-rank(ts_rank(close, {w})) * rank(volume / adv20)",
]
results = [engine.research(t, data=data, param_space={"w": [3, 5, 10, 20]}, max_evals=100) for t in templates]
book = engine.build_alpha_book(results, max_pairwise_corr=0.5, min_score=-10)
print(book.report())
