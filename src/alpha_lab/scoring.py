import math


def robust_fitness(metrics: dict) -> float:
    pf = metrics.get("profit_factor", 0.0)
    if math.isinf(pf):
        pf = 3.0
    return float(
        0.35 * metrics.get("net_sharpe", 0.0)
        + 0.15 * metrics.get("sortino", 0.0)
        + 0.10 * min(pf, 3.0)
        + 0.15 * metrics.get("rank_ic_mean", 0.0) * 100.0
        + 0.10 * metrics.get("calmar", 0.0)
        - 0.10 * abs(metrics.get("max_drawdown", 0.0)) * 10.0
        - 0.10 * metrics.get("turnover", 0.0)
        - 0.05 * metrics.get("cost_drag", 0.0) * 10.0
    )


def score_metrics(metrics: dict, objective="robust_fitness") -> float:
    if objective == "net_sharpe":
        return float(metrics.get("net_sharpe", 0.0))
    if objective == "net_return":
        return float(metrics.get("net_return", 0.0))
    if objective == "rank_ic":
        return float(metrics.get("rank_ic_mean", 0.0))
    if objective == "calmar":
        return float(metrics.get("calmar", 0.0))
    return robust_fitness(metrics)
