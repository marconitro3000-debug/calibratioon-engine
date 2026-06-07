import numpy as np
import pandas as pd
from alpha_lab.operators import rank, delta, ts_mean, scale


def test_basic_operators_shape():
    df = pd.DataFrame(np.random.randn(20, 5))
    assert rank(df).shape == df.shape
    assert delta(df, 2).shape == df.shape
    assert ts_mean(df, 5).shape == df.shape
    assert scale(df).abs().sum(axis=1).dropna().max() <= 1.0000001
