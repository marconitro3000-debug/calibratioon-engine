from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab import analyze_portfolio_risk


def make_demo_prices(path: Path) -> None:
    rng = np.random.default_rng(9)
    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    returns = rng.normal(
        loc=[0.0004, 0.0003, 0.0005, 0.0002],
        scale=[0.018, 0.015, 0.026, 0.010],
        size=(len(dates), 4),
    )
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        columns=["AAPL", "MSFT", "NVDA", "SPY"],
    )
    prices.insert(0, "date", dates)
    prices.to_csv(path, index=False)


def main() -> None:
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    holdings_path = data_dir / "portfolio_holdings.json"
    prices_path = data_dir / "portfolio_prices.csv"

    holdings_path.write_text(
        """
{
  "holdings": [
    {"symbol": "AAPL", "market_value": 12000},
    {"symbol": "MSFT", "market_value": 8000},
    {"symbol": "NVDA", "market_value": 5000}
  ]
}
""".strip(),
        encoding="utf-8",
    )
    make_demo_prices(prices_path)

    result = analyze_portfolio_risk(
        holdings=holdings_path,
        prices=prices_path,
        benchmark="SPY",
    )
    print(result.report())


if __name__ == "__main__":
    main()
