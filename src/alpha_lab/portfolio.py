from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from .metrics import annualized_sharpe, max_drawdown

PathLike: TypeAlias = str | Path
HoldingsInput: TypeAlias = dict[str, float] | pd.DataFrame | PathLike
PricesInput: TypeAlias = pd.DataFrame | PathLike


@dataclass
class PortfolioRiskResult:
    holdings: pd.Series
    weights: pd.Series
    metrics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    portfolio_returns: pd.Series | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "holdings": self.holdings.to_dict(),
            "weights": self.weights.to_dict(),
            "metrics": self.metrics,
            "warnings": self.warnings,
        }

    def report(self) -> str:
        lines = [
            "# Portfolio Risk Report",
            "",
            f"Portfolio value: `{self.metrics['portfolio_value']:.2f}`",
            "",
            "## Risk Metrics",
        ]
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.6f}")
            elif isinstance(value, dict):
                lines.append(f"- {key}: {value}")
            else:
                lines.append(f"- {key}: {value}")
        if self.warnings:
            lines += ["", "## Warnings"]
            lines += [f"- {warning}" for warning in self.warnings]
        return "\n".join(lines)


def load_holdings(source: HoldingsInput) -> pd.Series:
    if isinstance(source, dict):
        return _normalize_holdings_series(pd.Series(source, dtype=float))
    if isinstance(source, pd.DataFrame):
        return _holdings_from_frame(source)

    path = Path(source)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return _holdings_from_json(payload)
    if path.suffix.lower() == ".csv":
        return _holdings_from_frame(pd.read_csv(path))
    raise ValueError(f"unsupported holdings file format: {path.suffix}")


def load_prices(source: PricesInput) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        prices = source.copy()
    else:
        path = Path(source)
        if path.suffix.lower() == ".csv":
            prices = pd.read_csv(path)
        elif path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            prices = _prices_from_json(payload)
        else:
            raise ValueError(f"unsupported prices file format: {path.suffix}")

    return _normalize_prices(prices)


def analyze_portfolio_risk(
    holdings: HoldingsInput,
    prices: PricesInput,
    *,
    benchmark: str | None = None,
    var_level: float = 0.95,
) -> PortfolioRiskResult:
    return PortfolioRiskEngine(var_level=var_level).analyze(holdings, prices, benchmark=benchmark)


class PortfolioRiskEngine:
    def __init__(self, *, var_level: float = 0.95, periods_per_year: int = 252):
        if not 0 < var_level < 1:
            raise ValueError("var_level must be between 0 and 1")
        self.var_level = var_level
        self.periods_per_year = periods_per_year

    def analyze(
        self,
        holdings: HoldingsInput,
        prices: PricesInput,
        *,
        benchmark: str | None = None,
    ) -> PortfolioRiskResult:
        holdings_series = load_holdings(holdings)
        price_frame = load_prices(prices)
        missing = sorted(set(holdings_series.index) - set(price_frame.columns))
        if missing:
            raise ValueError(f"missing price columns for holdings: {missing}")

        portfolio_prices = price_frame.loc[:, holdings_series.index].ffill().dropna(how="all")
        returns = portfolio_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
        weights = holdings_series / holdings_series.sum()
        portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
        metrics = self._metrics(holdings_series, weights, returns, portfolio_returns, price_frame, benchmark)
        warnings = self._warnings(metrics)
        return PortfolioRiskResult(
            holdings=holdings_series,
            weights=weights,
            metrics=metrics,
            warnings=warnings,
            portfolio_returns=portfolio_returns,
        )

    def _metrics(
        self,
        holdings: pd.Series,
        weights: pd.Series,
        asset_returns: pd.DataFrame,
        portfolio_returns: pd.Series,
        prices: pd.DataFrame,
        benchmark: str | None,
    ) -> dict[str, Any]:
        var_tail = 1.0 - self.var_level
        var_value = float(portfolio_returns.quantile(var_tail)) if len(portfolio_returns) else 0.0
        cvar_sample = portfolio_returns[portfolio_returns <= var_value]
        corr = asset_returns.corr().replace([np.inf, -np.inf], np.nan)
        off_diag = corr.where(~np.eye(len(corr), dtype=bool)).stack()
        beta = self._beta(portfolio_returns, prices, benchmark)
        contribution = weights * asset_returns.std().reindex(weights.index).fillna(0.0)
        contribution = contribution / contribution.abs().sum() if contribution.abs().sum() else contribution

        return {
            "portfolio_value": float(holdings.sum()),
            "asset_count": int(len(holdings)),
            "daily_volatility": float(portfolio_returns.std()) if len(portfolio_returns) else 0.0,
            "annualized_volatility": (
                float(portfolio_returns.std() * np.sqrt(self.periods_per_year)) if len(portfolio_returns) else 0.0
            ),
            "annualized_return": _annualized_return(portfolio_returns, self.periods_per_year),
            "sharpe": annualized_sharpe(portfolio_returns, self.periods_per_year),
            "max_drawdown": max_drawdown(portfolio_returns),
            f"var_{int(self.var_level * 100)}": var_value,
            f"cvar_{int(self.var_level * 100)}": float(cvar_sample.mean()) if len(cvar_sample) else var_value,
            "best_day": float(portfolio_returns.max()) if len(portfolio_returns) else 0.0,
            "worst_day": float(portfolio_returns.min()) if len(portfolio_returns) else 0.0,
            "top_position_weight": float(weights.abs().max()) if len(weights) else 0.0,
            "herfindahl_concentration": float((weights**2).sum()) if len(weights) else 0.0,
            "average_pairwise_correlation": float(off_diag.mean()) if len(off_diag) else 0.0,
            "beta_to_benchmark": beta,
            "largest_position": str(weights.abs().idxmax()) if len(weights) else "",
            "risk_contribution": contribution.fillna(0.0).to_dict(),
            "weights": weights.to_dict(),
        }

    def _beta(self, portfolio_returns: pd.Series, prices: pd.DataFrame, benchmark: str | None) -> float | None:
        if not benchmark:
            return None
        if benchmark not in prices.columns:
            raise ValueError(f"benchmark column not found in prices: {benchmark}")
        benchmark_returns = prices[benchmark].pct_change().replace([np.inf, -np.inf], np.nan)
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 3:
            return None
        variance = aligned.iloc[:, 1].var()
        if variance == 0 or np.isnan(variance):
            return None
        return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / variance)

    def _warnings(self, metrics: dict[str, Any]) -> list[str]:
        warnings = []
        if metrics["top_position_weight"] > 0.35:
            warnings.append("High concentration: top position is above 35% of portfolio value.")
        if metrics["herfindahl_concentration"] > 0.25:
            warnings.append("Portfolio is concentrated across a small number of holdings.")
        if metrics["average_pairwise_correlation"] > 0.65:
            warnings.append("High average correlation between assets reduces diversification.")
        if metrics["annualized_volatility"] > 0.30:
            warnings.append("Annualized volatility is above 30%.")
        if metrics["max_drawdown"] < -0.25:
            warnings.append("Historical drawdown is worse than -25%.")
        return warnings


def _normalize_holdings_series(series: pd.Series) -> pd.Series:
    holdings = pd.to_numeric(series, errors="coerce").dropna()
    holdings.index = holdings.index.astype(str).str.upper()
    holdings = holdings.groupby(level=0).sum().sort_index()
    if holdings.empty:
        raise ValueError("holdings are empty")
    if holdings.sum() == 0:
        raise ValueError("holdings total value cannot be zero")
    return holdings.astype(float)


def _holdings_from_json(payload: Any) -> pd.Series:
    if isinstance(payload, dict) and "holdings" not in payload:
        return _normalize_holdings_series(pd.Series(payload, dtype=float))
    records = payload.get("holdings", payload) if isinstance(payload, dict) else payload
    return _holdings_from_frame(pd.DataFrame(records))


def _holdings_from_frame(frame: pd.DataFrame) -> pd.Series:
    columns = {str(col).lower(): col for col in frame.columns}
    symbol_col = _first_existing(columns, ["symbol", "ticker", "asset"])
    value_col = _first_existing(columns, ["market_value", "value", "amount", "notional"])
    if symbol_col is None:
        raise ValueError("holdings must include a symbol/ticker/asset column")
    if value_col is None:
        raise ValueError("holdings must include market_value/value/amount/notional")
    series = frame.set_index(symbol_col)[value_col]
    return _normalize_holdings_series(series)


def _prices_from_json(payload: Any) -> pd.DataFrame:
    if isinstance(payload, dict) and all(isinstance(value, dict | list) for value in payload.values()):
        frame = pd.DataFrame(payload)
        if "date" in frame.columns:
            return frame
        return frame.reset_index(names="date")
    return pd.DataFrame(payload)


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    date_col = next((col for col in frame.columns if str(col).lower() in {"date", "timestamp", "time"}), None)
    if date_col is not None:
        frame[date_col] = pd.to_datetime(frame[date_col])
        frame = frame.set_index(date_col)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    frame.columns = frame.columns.astype(str).str.upper()
    frame = frame.apply(pd.to_numeric, errors="coerce").sort_index()
    frame = frame.loc[:, ~frame.columns.duplicated()]
    if frame.empty:
        raise ValueError("prices are empty")
    return frame


def _first_existing(columns: dict[str, Any], names: list[str]) -> Any | None:
    for name in names:
        if name in columns:
            return columns[name]
    return None


def _annualized_return(returns: pd.Series, periods_per_year: int) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0
    return float((1 + returns).prod() ** (periods_per_year / len(returns)) - 1)
