import json

import numpy as np
import pandas as pd

from alpha_lab import PortfolioRiskEngine, analyze_portfolio_risk, load_holdings, load_prices


def make_prices() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    returns = rng.normal(0.0002, 0.012, size=(len(dates), 4))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        columns=["AAPL", "MSFT", "NVDA", "SPY"],
    )
    prices.insert(0, "date", dates)
    return prices


def test_load_holdings_from_json_and_csv(tmp_path):
    json_path = tmp_path / "holdings.json"
    csv_path = tmp_path / "holdings.csv"

    json_path.write_text(
        json.dumps(
            {
                "holdings": [
                    {"symbol": "AAPL", "market_value": 12000},
                    {"symbol": "MSFT", "market_value": 8000},
                ]
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"ticker": "AAPL", "value": 12000},
            {"ticker": "MSFT", "value": 8000},
        ]
    ).to_csv(csv_path, index=False)

    assert load_holdings(json_path).sum() == 20000
    assert load_holdings(csv_path).sum() == 20000


def test_load_prices_from_csv_and_json(tmp_path):
    prices = make_prices()
    csv_path = tmp_path / "prices.csv"
    json_path = tmp_path / "prices.json"
    prices.to_csv(csv_path, index=False)
    prices.assign(date=prices["date"].astype(str)).to_json(json_path, orient="records")

    assert load_prices(csv_path).shape == (80, 4)
    assert load_prices(json_path).shape == (80, 4)


def test_analyze_portfolio_risk_from_files(tmp_path):
    holdings_path = tmp_path / "holdings.json"
    prices_path = tmp_path / "prices.csv"
    holdings_path.write_text(json.dumps({"AAPL": 12000, "MSFT": 8000, "NVDA": 5000}), encoding="utf-8")
    make_prices().to_csv(prices_path, index=False)

    result = analyze_portfolio_risk(holdings_path, prices_path, benchmark="SPY")

    assert result.metrics["portfolio_value"] == 25000
    assert result.metrics["asset_count"] == 3
    assert "annualized_volatility" in result.metrics
    assert "var_95" in result.metrics
    assert result.metrics["beta_to_benchmark"] is not None
    assert "Portfolio Risk Report" in result.report()


def test_portfolio_engine_accepts_dict_and_dataframe():
    result = PortfolioRiskEngine().analyze(
        holdings={"AAPL": 12000, "MSFT": 8000},
        prices=make_prices(),
    )

    assert round(float(result.weights.sum()), 8) == 1.0
    assert result.portfolio_returns is not None
