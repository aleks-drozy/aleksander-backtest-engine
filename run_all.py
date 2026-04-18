#!/usr/bin/env python3
"""CLI: fetch data, run all strategies, validate output, write results JSON."""
import json
from datetime import datetime, timezone
from pathlib import Path

from data.fetcher import fetch_ohlcv
from engine.backtester import Backtester
from schemas import BacktestResults
from strategies.ifvg_cisd import IFVGCISDStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.sma_crossover import SMACrossoverStrategy

START = "2018-01-01"
END = "2025-12-31"

REGISTRY = [
    (SMACrossoverStrategy(), "SPY"),
    (RSIMeanReversionStrategy(), "SPY"),
    (IFVGCISDStrategy(), "NQ=F"),
]


def main() -> None:
    strategy_results = []
    asset_universe: dict[str, str] = {}

    for strategy, ticker in REGISTRY:
        print(f"Running {strategy.name} on {ticker}...")
        df = fetch_ohlcv(ticker, START, END)
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
