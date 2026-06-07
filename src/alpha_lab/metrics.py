import numpy as np
import pandas as pd


def annualized_sharpe(r: pd.Series, periods=252) -> float:
    r = pd.Series(r).dropna()
    sd = r.std()
    if len(r) < 3 or sd == 0 or np.isnan(sd):
        return 0.0
    return float(np.sqrt(periods) * r.mean() / sd)


def sortino(r: pd.Series, periods=252) -> float:
    r = pd.Series(r).dropna()
    downside = r[r < 0].std()
    if len(r) < 3 or downside == 0 or np.isnan(downside):
        return 0.0
    return float(np.sqrt(periods) * r.mean() / downside)


def max_drawdown(r: pd.Series) -> float:
    curve = (1 + pd.Series(r).fillna(0)).cumprod()
    dd = curve / curve.cummax() - 1
    return float(dd.min())


def calmar(r: pd.Series) -> float:
    mdd = abs(max_drawdown(r))
    if mdd == 0:
        return 0.0
    ann = (1 + pd.Series(r).fillna(0)).prod() ** (252 / max(len(r), 1)) - 1
    return float(ann / mdd)


def profit_factor(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def hit_rate(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    return float((r > 0).mean()) if len(r) else 0.0


def turnover(positions: pd.DataFrame) -> float:
    return float(positions.diff().abs().sum(axis=1).mean())


def information_coefficient(signal: pd.DataFrame, fwd_returns: pd.DataFrame, method="pearson") -> pd.Series:
    out = []
    idx = []
    for dt in signal.index.intersection(fwd_returns.index):
        s = signal.loc[dt].replace([np.inf, -np.inf], np.nan)
        r = fwd_returns.loc[dt].replace([np.inf, -np.inf], np.nan)
        df = pd.concat([s, r], axis=1).dropna()
        if len(df) < 3:
            out.append(np.nan)
        else:
            out.append(df.iloc[:, 0].corr(df.iloc[:, 1], method=method))
        idx.append(dt)
    return pd.Series(out, index=idx)


def decay_curve(signal: pd.DataFrame, close: pd.DataFrame, horizons=(1, 2, 3, 5, 10), method="spearman") -> dict:
    out = {}
    for h in horizons:
        fwd = close.pct_change(int(h)).shift(-int(h))
        ic = information_coefficient(signal, fwd, method=method)
        out[int(h)] = float(ic.mean(skipna=True)) if len(ic) else 0.0
    return out


def summarize(bt: dict, signal: pd.DataFrame, data: dict) -> dict:
    gross = bt["gross_returns"]
    net = bt["net_returns"]
    pos = bt["positions"]
    fwd = data["close"].pct_change().shift(-1)
    ic = information_coefficient(signal, fwd, method="pearson")
    ric = information_coefficient(signal, fwd, method="spearman")
    ic_std = ic.std(skipna=True)
    ic_mean = float(ic.mean(skipna=True)) if len(ic) else 0.0
    rank_ic_mean = float(ric.mean(skipna=True)) if len(ric) else 0.0
    icir = float(ic_mean / ic_std) if ic_std and not np.isnan(ic_std) else 0.0
    cost_drag = float(bt["costs"].sum())
    avg_daily = float(net.mean()) if len(net) else 0.0
    worst_day = float(net.min()) if len(net) else 0.0
    best_day = float(net.max()) if len(net) else 0.0
    return {
        "gross_return": float(gross.sum()),
        "net_return": float(net.sum()),
        "gross_sharpe": annualized_sharpe(gross),
        "net_sharpe": annualized_sharpe(net),
        "sortino": sortino(net),
        "calmar": calmar(net),
        "max_drawdown": max_drawdown(net),
        "profit_factor": profit_factor(net),
        "hit_rate": hit_rate(net),
        "turnover": turnover(pos),
        "cost_drag": cost_drag,
        "average_daily_pnl": avg_daily,
        "worst_day": worst_day,
        "best_day": best_day,
        "ic_mean": ic_mean,
        "rank_ic_mean": rank_ic_mean,
        "ic_ir": icir,
        "IC": ic_mean,
        "rank_IC": rank_ic_mean,
        "ICIR": icir,
        "alpha_decay": decay_curve(signal, data["close"]),
        "parameter_stability": 0.0,
        "oos_sharpe": 0.0,
        "oos_degradation": 0.0,
        "deflated_sharpe_proxy": 0.0,
        "pbo_proxy": 0.0,
        "fitness": 0.0,
    }
