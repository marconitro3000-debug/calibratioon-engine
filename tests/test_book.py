import numpy as np
import pandas as pd

from alpha_lab import AutoAlphaResearchEngine


def test_alpha_book_runs():
    rng = np.random.default_rng(5)
    n, m = 160, 10
    rets = rng.normal(0, 0.01, size=(n, m))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), columns=[f"A{i}" for i in range(m)])
    volume = pd.DataFrame(rng.lognormal(10, 0.2, size=(n, m)), columns=close.columns)
    data = {"close": close, "volume": volume}
    engine = AutoAlphaResearchEngine(neutralization=["market"])
    r1 = engine.test_alpha("-rank(delta(close, 3))", data)
    r2 = engine.test_alpha("rank(ts_corr(close, volume, 10))", data)
    book = engine.build_alpha_book([r1, r2], min_score=-999)
    assert book.summary["selected"] >= 1
