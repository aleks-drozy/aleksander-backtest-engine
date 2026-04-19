#!/usr/bin/env python3
"""CLI: fetch data, run all strategies, validate output, write results JSON."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data.fetcher import fetch_ohlcv
from engine.backtester import Backtester
from schemas import BacktestResults
from strategies.ifvg_cisd import IFVGCISDStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.sma_crossover import SMACrossoverStrategy

# 15m data on yfinance has a rolling 60-day window; compute start dynamically
INTERVAL = "15m"
END = datetime.now(timezone.utc).strftime("%Y-%m-%d")
START = (datetime.now(timezone.utc) - timedelta(days=58)).strftime("%Y-%m-%d")

REGISTRY = [
    (SMACrossoverStrategy(), "NQ=F"),
    (RSIMeanReversionStrategy(), "NQ=F"),
    (IFVGCISDStrategy(), "NQ=F"),
]


def main() -> None:
    strategy_results = []
    asset_universe: dict[str, str] = {}

    for strategy, ticker in REGISTRY:
        print(f"Running {strategy.name} on {ticker} [{INTERVAL}]...")
        df = fetch_ohlcv(ticker, START, END, INTERVAL)
        result = Backtester(strategy).run(df)
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
    out_path.write_text(validated.model_dump_json(indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
