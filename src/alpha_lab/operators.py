import numpy as np
import pandas as pd


def _as_df(x):
    if isinstance(x, pd.DataFrame):
        return x
    raise TypeError("operator input must be a pandas DataFrame")


def rank(x: pd.DataFrame) -> pd.DataFrame:
    x = _as_df(x)
    return x.rank(axis=1, pct=True) - 0.5


def zscore(x: pd.DataFrame) -> pd.DataFrame:
    x = _as_df(x)
    mu = x.mean(axis=1)
    sd = x.std(axis=1).replace(0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)


def scale(x: pd.DataFrame, target: float = 1.0) -> pd.DataFrame:
    x = _as_df(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = x.abs().sum(axis=1).replace(0, np.nan)
    return x.div(gross, axis=0).fillna(0.0) * target


def delay(x: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return _as_df(x).shift(int(n))


def delta(x: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return _as_df(x).diff(int(n))


def returns(x: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return _as_df(x).pct_change(int(n)).replace([np.inf, -np.inf], np.nan)


def ts_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return _as_df(x).rolling(int(n), min_periods=max(2, int(n)//3)).mean()


def ts_std(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return _as_df(x).rolling(int(n), min_periods=max(2, int(n)//3)).std()


def ts_min(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return _as_df(x).rolling(int(n), min_periods=max(2, int(n)//3)).min()


def ts_max(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return _as_df(x).rolling(int(n), min_periods=max(2, int(n)//3)).max()


def ts_rank(x: pd.DataFrame, n: int) -> pd.DataFrame:
    x = _as_df(x)
    n = int(n)
    return x.rolling(n, min_periods=max(2, n//3)).apply(lambda a: pd.Series(a).rank(pct=True).iloc[-1] - 0.5, raw=False)


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return _as_df(x).rolling(int(n), min_periods=max(3, int(n)//3)).corr(_as_df(y))


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return _as_df(x).rolling(int(n), min_periods=max(3, int(n)//3)).cov(_as_df(y))


def decay_linear(x: pd.DataFrame, n: int) -> pd.DataFrame:
    x = _as_df(x)
    n = int(n)
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()
    return x.rolling(n, min_periods=max(2, n//3)).apply(lambda a: np.dot(a[-len(weights):], weights[-len(a):]), raw=True)


def signed_power(x: pd.DataFrame, p: float) -> pd.DataFrame:
    x = _as_df(x)
    return np.sign(x) * (x.abs() ** float(p))


def winsorize(x: pd.DataFrame, limit: float = 0.05) -> pd.DataFrame:
    x = _as_df(x).copy()
    lo = x.quantile(float(limit), axis=1)
    hi = x.quantile(1.0 - float(limit), axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


def neutralize(x: pd.DataFrame) -> pd.DataFrame:
    x = _as_df(x)
    return x.sub(x.mean(axis=1), axis=0)


def group_neutralize(x: pd.DataFrame, groups: dict | None = None) -> pd.DataFrame:
    x = _as_df(x).copy()
    if not groups:
        return neutralize(x)
    out = x.copy()
    by_group = {}
    for asset in x.columns:
        by_group.setdefault(groups.get(asset, "__ungrouped__"), []).append(asset)
    for cols in by_group.values():
        out[cols] = x[cols].sub(x[cols].mean(axis=1), axis=0)
    return out
