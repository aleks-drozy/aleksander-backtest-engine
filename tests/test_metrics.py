import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_df
from engine.metrics import compute_metrics, sample_equity_weekly


def _always_long_returns(n: int = 100, daily_return: float = 0.001) -> tuple:
    df = make_df(n, daily_return=daily_return)
    pos = pd.Series(1.0, index=df.index)
    pct_ret = df["Close"].pct_change().fillna(0)
    net_returns = pos * pct_ret
    return net_returns, pos, df["Close"]


def test_compute_metrics_returns_required_keys():
    net, pos, prices = _always_long_returns()
    result = compute_metrics(net, pos, prices)
    assert "metrics" in result
    m = result["metrics"]
    for key in ("total_return_pct", "sharpe", "max_drawdown_pct", "win_rate_pct", "profit_factor", "num_trades"):
        assert key in m, f"Missing key: {key}"


def test_total_return_positive_for_uptrend():
    net, pos, prices = _always_long_returns(daily_return=0.001)
    result = compute_metrics(net, pos, prices)
    assert result["metrics"]["total_return_pct"] > 0


def test_max_drawdown_is_negative_or_zero():
    net, pos, prices = _always_long_returns()
    result = compute_metrics(net, pos, prices)
    assert result["metrics"]["max_drawdown_pct"] <= 0


def test_num_trades_counted():
    df = make_df(20)
    pos = pd.Series([0] * 5 + [1] * 10 + [0] * 5, index=df.index, dtype=float)
    pct_ret = df["Close"].pct_change().fillna(0)
    net = pos * pct_ret
    result = compute_metrics(net, pos, df["Close"])
    assert result["metrics"]["num_trades"] == 1


def test_sample_equity_weekly_reduces_points():
    df = make_df(300)
    equity = pd.Series(
        10_000 * (1.001 ** np.arange(300)),
        index=df.index,
    )
    weekly = sample_equity_weekly(equity)
    assert len(weekly) < 300
    assert weekly[0]["value"] == pytest.approx(10_000, rel=0.05)
    assert all("date" in p and "value" in p for p in weekly)


def test_sample_equity_weekly_format():
    df = make_df(50)
    equity = pd.Series(10_000.0, index=df.index)
    weekly = sample_equity_weekly(equity)
    assert isinstance(weekly, list)
    assert isinstance(weekly[0]["date"], str)
    assert isinstance(weekly[0]["value"], float)
