#!/usr/bin/env python3
"""CLI: fetch data, run all strategies, validate output, write results JSON."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from data.fetcher import fetch_ohlcv
from engine.backtester import Backtester
from schemas import BacktestResults
from strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from strategies.ifvg_cisd import IFVGCISDStrategy
from strategies.macd_bollinger_combo import MACDBollingerComboStrategy
from strategies.macd_crossover import MACDCrossoverStrategy
from strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.sma_crossover import SMACrossoverStrategy

# yfinance rolling window limits per interval
INTERVAL_WINDOW_DAYS = {"1h": 720, "30m": 58, "15m": 58}

END = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# All strategies on 1h — 720 days of data gives a meaningful OOS window (~216 days)
# Parameters have been rescaled for hourly bars throughout the strategy files
REGISTRY = [
    (SMACrossoverStrategy(),           "NQ=F", "1h"),
    (RSIMeanReversionStrategy(),       "NQ=F", "1h"),
    (IFVGCISDStrategy(),               "NQ=F", "1h"),
    (MACDCrossoverStrategy(),          "NQ=F", "1h"),
    (BollingerMeanReversionStrategy(), "NQ=F", "1h"),
    (MACDBollingerComboStrategy(),     "NQ=F", "1h"),
    (OpeningRangeBreakoutStrategy(),   "NQ=F", "1h"),
]


def filter_session(df: pd.DataFrame, start_time: str = "09:30", end_time: str = "16:00") -> pd.DataFrame:
    """
    Restrict intraday data to the US cash session (9:30–16:00 ET).
    Eliminates thin overnight/Asian session bars that generate noisy signals.
    """
    idx = df.index
    if idx.tzinfo is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("America/New_York")
    df = df.copy()
    df.index = idx
    df = df.between_time(start_time, end_time)
    # Drop tz info after filtering so downstream code stays timezone-naive
    df.index = df.index.tz_localize(None)
    return df


def main() -> None:
    strategy_results = []
    asset_universe: dict[str, str] = {}

    for strategy, ticker, interval in REGISTRY:
        window = INTERVAL_WINDOW_DAYS.get(interval, 58)
        start = (datetime.now(timezone.utc) - timedelta(days=window)).strftime("%Y-%m-%d")
        print(f"Running {strategy.name} on {ticker} [{interval}]...")
        df = fetch_ohlcv(ticker, start, END, interval)
        df = filter_session(df)   # US cash session only: 09:30–16:00 ET
        result = Backtester(strategy).run(df)
        result["timeframe"] = interval
        strategy_results.append(result)
        asset_universe[result["id"]] = ticker

    # Benchmark: NQ buy-and-hold over the OOS window (last 30% of session bars)
    bm_window = INTERVAL_WINDOW_DAYS.get("1h", 720)
    bm_start = (datetime.now(timezone.utc) - timedelta(days=bm_window)).strftime("%Y-%m-%d")
    bm_df = filter_session(fetch_ohlcv("NQ=F", bm_start, END, "1h"))
    split_idx = int(len(bm_df) * 0.70)
    oos_prices = bm_df["Close"].iloc[split_idx:]
    benchmark_return_pct = round(
        (float(oos_prices.iloc[-1]) / float(oos_prices.iloc[0]) - 1) * 100, 2
    )
    print(f"Benchmark (NQ buy-and-hold OOS): {benchmark_return_pct:+.2f}%")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset_universe": asset_universe,
        "benchmark_return_pct": benchmark_return_pct,
        "strategies": strategy_results,
    }

    validated = BacktestResults(**output)

    out_path = Path("results/backtest_results.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(validated.model_dump_json(indent=2), encoding="utf-8")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
