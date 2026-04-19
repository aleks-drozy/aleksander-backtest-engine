import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_df


# ── SMA Crossover ──────────────────────────────────────────────────────────────

def test_sma_crossover_signals_valid_values():
    from strategies.sma_crossover import SMACrossoverStrategy
    df = make_df(200)
    s = SMACrossoverStrategy()
    signals = s.generate_signals(df)
    assert set(signals.dropna().unique()).issubset({-1.0, 0.0, 1.0})
    assert len(signals) == len(df)


def test_sma_crossover_supports_both_directions():
    from strategies.sma_crossover import SMACrossoverStrategy
    s = SMACrossoverStrategy()
    assert s.direction == "both"


def test_sma_crossover_golden_cross_produces_long():
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = np.concatenate([
        100 - np.arange(60) * 0.5,
        70 + np.arange(140) * 0.6,
    ])
    df = pd.DataFrame(
        {"Open": prices, "High": prices * 1.01, "Low": prices * 0.99, "Close": prices},
        index=dates,
    )
    from strategies.sma_crossover import SMACrossoverStrategy
    signals = SMACrossoverStrategy().generate_signals(df)
    assert (signals == 1).any()


# ── RSI Mean Reversion ─────────────────────────────────────────────────────────

def test_rsi_mean_reversion_signals_valid_values():
    from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
    df = make_df(200)
    s = RSIMeanReversionStrategy()
    signals = s.generate_signals(df)
    assert set(signals.dropna().unique()).issubset({-1.0, 0.0, 1.0})
    assert len(signals) == len(df)


def test_rsi_mean_reversion_supports_both_directions():
    from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
    s = RSIMeanReversionStrategy()
    assert s.direction == "both"


def test_rsi_mean_reversion_fires_on_oversold():
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = np.concatenate([
        np.linspace(100, 40, 50),
        np.linspace(40, 80, 50),
    ])
    df = pd.DataFrame(
        {"Open": prices, "High": prices * 1.005, "Low": prices * 0.995, "Close": prices},
        index=dates,
    )
    from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
    signals = RSIMeanReversionStrategy().generate_signals(df)
    assert (signals == 1).any(), "Expected at least one long entry after oversold RSI"


# ── IFVG + CISD ────────────────────────────────────────────────────────────────

def test_ifvg_cisd_signals_valid_values():
    from strategies.ifvg_cisd import IFVGCISDStrategy
    df = make_df(300)
    s = IFVGCISDStrategy()
    signals = s.generate_signals(df)
    assert set(signals.dropna().unique()).issubset({-1.0, 0.0, 1.0})
    assert len(signals) == len(df)


def test_ifvg_cisd_supports_both_directions():
    from strategies.ifvg_cisd import IFVGCISDStrategy
    s = IFVGCISDStrategy()
    assert s.direction == "both"


def test_ifvg_cisd_generates_some_signals_on_volatile_data():
    rng = np.random.default_rng(42)
    n = 400
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    returns = rng.normal(0, 0.015, n)
    prices = 100 * np.exp(np.cumsum(returns))
    highs = prices * (1 + np.abs(rng.normal(0, 0.005, n)))
    lows = prices * (1 - np.abs(rng.normal(0, 0.005, n)))
    df = pd.DataFrame(
        {"Open": prices, "High": highs, "Low": lows, "Close": prices},
        index=dates,
    )
    from strategies.ifvg_cisd import IFVGCISDStrategy
    signals = IFVGCISDStrategy().generate_signals(df)
    assert (signals != 0).any(), "Expected some non-zero signals on volatile data"
