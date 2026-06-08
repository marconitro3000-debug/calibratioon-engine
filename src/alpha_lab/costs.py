import pandas as pd


def transaction_costs(
    positions: pd.DataFrame, *, commission_bps=0.0, slippage_bps=0.0, spread_bps=0.0, borrow_bps=0.0
) -> pd.Series:
    turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
    trading_bps = float(commission_bps) + float(slippage_bps) + float(spread_bps)
    trading_cost = turnover * trading_bps / 10000.0
    short_exposure = positions.clip(upper=0).abs().sum(axis=1)
    borrow_cost = short_exposure * float(borrow_bps) / 10000.0 / 252.0
    return trading_cost + borrow_cost


def cost_drag(gross_returns: pd.Series, net_returns: pd.Series) -> float:
    return float((gross_returns - net_returns).sum())
