#!/usr/bin/env python3
"""CLI: fetch data, run all strategies, validate output, write results JSON."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data.fetcher import fetch_ohlcv
from engine.backtester import Backtester
from schemas import BacktestResults
from strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from strategies.ifvg_cisd import IFVGCISDStrategy
from strategies.macd_bollinger_combo import MACDBollingerComboStrategy
from strategies.macd_crossover import MACDCrossoverStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.sma_crossover import SMACrossoverStrategy

# yfinance rolling window limits per interval
INTERVAL_WINDOW_DAYS = {"1h": 720, "30m": 58, "15m": 58}

END = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# All strategies on 1h — 720 days of data gives a meaningful OOS window (~216 days)
# Parameters have been rescaled for hourly bars throughout the strategy files
REGISTRY = [
    (SMACrossoverStrategy(),          "NQ=F", "1h"),
    (RSIMeanReversionStrategy(),      "NQ=F", "1h"),
    (IFVGCISDStrategy(),              "NQ=F", "1h"),
    (MACDCrossoverStrategy(),         "NQ=F", "1h"),
    (BollingerMeanReversionStrategy(), "NQ=F", "1h"),
    (MACDBollingerComboStrategy(),    "NQ=F", "1h"),
]


def main() -> None:
    strategy_results = []
    asset_universe: dict[str, str] = {}

    for strategy, ticker, interval in REGISTRY:
        window = INTERVAL_WINDOW_DAYS.get(interval, 58)
        start = (datetime.now(timezone.utc) - timedelta(days=window)).strftime("%Y-%m-%d")
        print(f"Running {strategy.name} on {ticker} [{interval}]...")
        df = fetch_ohlcv(ticker, start, END, interval)
        result = Backtester(strategy).run(df)
        result["timeframe"] = interval
        strategy_results.append(result)
        asset_universe[result["id"]] = ticker

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset_universe": asset_universe,
        "strategies": strategy_results,
    }

    validated = BacktestResults(**output)

    out_path = Path("results/backtest_results.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(validated.model_dump_json(indent=2), encoding="utf-8")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
