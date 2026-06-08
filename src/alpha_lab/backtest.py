import numpy as np
import pandas as pd

from .costs import transaction_costs
from .operators import decay_linear, group_neutralize, neutralize, scale, winsorize


def signal_to_positions(
    signal: pd.DataFrame, *, delay_periods=1, neutralization=None, groups=None, gross=1.0, cap=None, decay=1
) -> pd.DataFrame:
    sig = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sig = winsorize(sig, 0.02)
    if neutralization:
        if "sector" in neutralization or "group" in neutralization:
            sig = group_neutralize(sig, groups=groups)
        if "market" in neutralization:
            sig = neutralize(sig)
    if decay and int(decay) > 1:
        sig = decay_linear(sig, int(decay)).fillna(0.0)
    if cap is not None:
        sig = sig.clip(lower=-float(cap), upper=float(cap))
    pos = scale(sig, target=float(gross)).shift(int(delay_periods)).fillna(0.0)
    return pos


def backtest_alpha(
    signal: pd.DataFrame,
    data: dict,
    *,
    costs=None,
    neutralization=None,
    groups=None,
    delay_periods=1,
    gross=1.0,
    cap=None,
    decay=1,
):
    if "close" not in data:
        raise KeyError("data must contain 'close'")
    fwd_ret = data["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    positions = signal_to_positions(
        signal,
        delay_periods=delay_periods,
        neutralization=neutralization,
        groups=groups,
        gross=gross,
        cap=cap,
        decay=decay,
    )
    asset_pnl = positions.shift(1).fillna(0.0) * fwd_ret
    gross_returns = asset_pnl.sum(axis=1).fillna(0.0)
    costs = costs or {}
    cost_series = transaction_costs(positions, **costs)
    net_returns = gross_returns - cost_series
    return {
        "positions": positions,
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "costs": cost_series,
        "asset_pnl": asset_pnl,
    }
