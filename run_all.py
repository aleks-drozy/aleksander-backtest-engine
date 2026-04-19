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

# (strategy, ticker, interval) — each strategy runs on its research-optimal timeframe
REGISTRY = [
    (SMACrossoverStrategy(),         "NQ=F", "1h"),   # trend-following needs clean trends
    (RSIMeanReversionStrategy(),     "NQ=F", "30m"),  # sweet spot: active but lower noise
    (IFVGCISDStrategy(),             "NQ=F", "15m"),  # ICT concepts native to 15m
    (MACDCrossoverStrategy(),        "NQ=F", "1h"),   # profit factor 1.21 at 1h vs 1.14 at 30m
    (BollingerMeanReversionStrategy(),"NQ=F","15m"),  # mean reversion viable at 15m
    (MACDBollingerComboStrategy(),   "NQ=F", "15m"),  # dual confirmation handles 15m noise
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
    out_path.write_text(validated.model_dump_json(indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
