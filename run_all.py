#!/usr/bin/env python3
"""CLI: fetch data, run all strategies, validate output, write results JSON."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data.fetcher import fetch_ohlcv
from engine.backtester import Backtester
from schemas import BacktestResults
from strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from strategies.ifvg_cisd import IFVGCISDStrategy
from strategies.macd_bollinger_combo import MACDBollingerComboStrategy
from strategies.macd_crossover import MACDCrossoverStrategy
from strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from strategies.overnight_gap_fade import OvernightGapFadeStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.vwap_mean_reversion import VWAPMeanReversionStrategy

# yfinance rolling window limits per interval
INTERVAL_WINDOW_DAYS = {"1h": 720, "30m": 58, "15m": 58}

END = datetime.now(timezone.utc).strftime("%Y-%m-%d")

REGISTRY = [
    (SMACrossoverStrategy(),           "NQ=F", "1h"),
    (RSIMeanReversionStrategy(),       "NQ=F", "1h"),
    (IFVGCISDStrategy(),               "NQ=F", "1h"),
    (MACDCrossoverStrategy(),          "NQ=F", "1h"),
    (BollingerMeanReversionStrategy(), "NQ=F", "1h"),
    (MACDBollingerComboStrategy(),     "NQ=F", "1h"),
    (OpeningRangeBreakoutStrategy(),   "NQ=F", "1h"),
    (VWAPMeanReversionStrategy(),      "NQ=F", "1h"),
    (OvernightGapFadeStrategy(),       "NQ=F", "1h"),
]


# ── Session filter ─────────────────────────────────────────────────────────────

def filter_session(df: pd.DataFrame, start_time: str = "09:30", end_time: str = "16:00") -> pd.DataFrame:
    """Restrict intraday data to the US cash session (9:30–16:00 ET)."""
    idx = df.index
    if idx.tzinfo is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("America/New_York")
    df = df.copy()
    df.index = idx
    df = df.between_time(start_time, end_time)
    df.index = df.index.tz_localize(None)
    return df


# ── VIX overlay ────────────────────────────────────────────────────────────────

def fetch_vix_daily(start: str, end: str) -> pd.Series:
    """
    Return a daily VIX series (^VIX close prices) indexed by date.
    Falls back to a neutral value (20.0) if the fetch fails.
    """
    try:
        vix_df = fetch_ohlcv("^VIX", start, end, "1d")
        vix = vix_df["Close"].rename("VIX")
        vix.index = pd.to_datetime(vix.index).normalize()
        return vix
    except Exception as exc:
        print(f"  [warn] VIX fetch failed ({exc}); defaulting to 20.0")
        dates = pd.date_range(start, end, freq="B")
        return pd.Series(20.0, index=dates, name="VIX")


def merge_vix(df: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """
    Merge daily VIX into an intraday OHLCV DataFrame.
    Each intraday bar gets the VIX close of that calendar date (forward-filled).
    """
    df = df.copy()
    bar_dates = df.index.normalize()
    df["VIX"] = bar_dates.map(vix).astype(float)
    df["VIX"] = df["VIX"].ffill().fillna(20.0)
    return df


# ── Parameter sensitivity ──────────────────────────────────────────────────────

def compute_param_sensitivity(strategy, df: pd.DataFrame, n_values: int = 5) -> list[dict]:
    """
    Grid-search ±20% around each numerical param.
    Returns a list of {param, values, oos_sharpes} dicts (one per param).
    Temporarily overrides strategy.params; always restores the original.
    """
    base_params = dict(strategy.params)
    results = []

    for param_name, base_val in base_params.items():
        if not isinstance(base_val, (int, float)) or base_val == 0:
            continue

        # Build a symmetric grid: 0.80×, 0.90×, 1.00×, 1.10×, 1.20×
        multipliers = np.linspace(0.8, 1.2, n_values)
        test_values: list[float] = []
        sharpes: list[float] = []

        for mult in multipliers:
            new_val = base_val * mult
            # Preserve int type where the original param was an int
            new_val_typed: float | int = int(round(new_val)) if isinstance(base_val, int) else round(new_val, 4)
            test_values.append(float(new_val_typed))

            # Temporarily override the param
            strategy.params = {**base_params, param_name: new_val_typed}
            try:
                result = Backtester(strategy).run(df)
                sharpes.append(result["out_of_sample"]["metrics"]["sharpe"])
            except Exception:
                sharpes.append(0.0)

        # Restore original params
        strategy.params = base_params

        results.append({
            "param":       param_name,
            "values":      test_values,
            "oos_sharpes": [round(s, 2) for s in sharpes],
        })

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    strategy_results = []
    asset_universe: dict[str, str] = {}

    # Fetch VIX once for the full window
    window = INTERVAL_WINDOW_DAYS.get("1h", 720)
    global_start = (datetime.now(timezone.utc) - timedelta(days=window)).strftime("%Y-%m-%d")
    vix_series = fetch_vix_daily(global_start, END)

    for strategy, ticker, interval in REGISTRY:
        win = INTERVAL_WINDOW_DAYS.get(interval, 58)
        start = (datetime.now(timezone.utc) - timedelta(days=win)).strftime("%Y-%m-%d")
        print(f"Running {strategy.name} on {ticker} [{interval}]...")

        df = fetch_ohlcv(ticker, start, END, interval)
        df = filter_session(df)
        df = merge_vix(df, vix_series)

        result = Backtester(strategy).run(df)
        result["timeframe"] = interval
        asset_universe[result["id"]] = ticker

        # Param sensitivity (runs several backtests per strategy)
        print(f"  Computing param sensitivity for {strategy.name}...")
        result["param_sensitivity"] = compute_param_sensitivity(strategy, df)

        strategy_results.append(result)

    # ── Benchmark: NQ buy-and-hold over the OOS window ──────────────────────
    bm_df = filter_session(fetch_ohlcv("NQ=F", global_start, END, "1h"))
    split_idx = int(len(bm_df) * 0.70)
    oos_prices = bm_df["Close"].iloc[split_idx:]
    benchmark_return_pct = round(
        (float(oos_prices.iloc[-1]) / float(oos_prices.iloc[0]) - 1) * 100, 2
    )
    print(f"Benchmark (NQ buy-and-hold OOS): {benchmark_return_pct:+.2f}%")

    # ── Strategy correlation matrix ──────────────────────────────────────────
    # Build daily P&L from OOS returns, compute pairwise Pearson correlation.
    pnl_dict: dict[str, pd.Series] = {}
    for r in strategy_results:
        oos_ret = r.pop("_oos_returns", None)   # remove internal field
        if oos_ret is not None and len(oos_ret) > 0:
            # Resample to daily so different strategies on same ticker align
            daily = oos_ret.resample("B").sum()
            pnl_dict[r["id"]] = daily

    corr_matrix: dict[str, dict[str, float]] = {}
    if pnl_dict:
        pnl_df = pd.DataFrame(pnl_dict).dropna(how="all")
        corr = pnl_df.corr().round(3)
        corr_matrix = {
            col: {row: float(corr.loc[row, col]) for row in corr.index}
            for col in corr.columns
        }

    # ── Assemble + validate + write ──────────────────────────────────────────
    output = {
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "asset_universe":     asset_universe,
        "benchmark_return_pct": benchmark_return_pct,
        "strategies":         strategy_results,
        "correlation_matrix": corr_matrix,
    }

    validated = BacktestResults(**output)

    out_path = Path("results/backtest_results.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(validated.model_dump_json(indent=2), encoding="utf-8")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
