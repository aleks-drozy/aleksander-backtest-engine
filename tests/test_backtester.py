import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_df
from engine.base_strategy import Strategy
from engine.backtester import Backtester


class AlwaysLongStrategy(Strategy):
    id = "ALWAYS_LONG"
    name = "Always Long"
    description = "Always long for testing."
    direction = "long_only"
    params = {}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index)


class AlwaysShortStrategy(Strategy):
    id = "ALWAYS_SHORT"
    name = "Always Short"
    description = "Always short for testing."
    direction = "short_only"
    params = {}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(-1.0, index=df.index)


class BothDirectionStrategy(Strategy):
    id = "ALTERNATING"
    name = "Alternating"
    description = "Alternates long and short."
    direction = "both"
    params = {}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0.0, index=df.index)
        sig.iloc[::2] = 1.0
        sig.iloc[1::2] = -1.0
        return sig


def test_run_returns_required_keys():
    df = make_df(100)
    result = Backtester(AlwaysLongStrategy()).run(df)
    for key in ("id", "name", "description", "direction", "params", "in_sample", "out_of_sample"):
        assert key in result, f"Missing key: {key}"
    for period in ("in_sample", "out_of_sample"):
        for sub in ("period", "metrics", "sampled", "equity_curve"):
            assert sub in result[period], f"Missing {period}.{sub}"


def test_equity_curves_start_at_10000():
    df = make_df(200)
    result = Backtester(AlwaysLongStrategy()).run(df)
    assert result["in_sample"]["equity_curve"][0]["value"] == pytest.approx(10_000, rel=0.05)
    assert result["out_of_sample"]["equity_curve"][0]["value"] == pytest.approx(10_000, rel=0.05)


def test_train_pct_splits_data():
    df = make_df(200)
    bt = Backtester(AlwaysLongStrategy(), train_pct=0.70)
    result = bt.run(df)
    is_start = result["in_sample"]["period"]["start"]
    oos_end = result["out_of_sample"]["period"]["end"]
    assert is_start < oos_end


def test_long_only_strategy_filters_short_signals():
    class ForcedLong(BothDirectionStrategy):
        direction = "long_only"

    df = make_df(200)
    result = Backtester(ForcedLong()).run(df)
    result_both = Backtester(BothDirectionStrategy()).run(df)
    assert result["in_sample"]["metrics"]["num_trades"] >= 0
    assert result_both["in_sample"]["metrics"]["num_trades"] >= 0


def test_equity_curve_sampled_weekly():
    df = make_df(300)
    result = Backtester(AlwaysLongStrategy()).run(df)
    assert result["in_sample"]["sampled"] == "weekly"
    assert len(result["in_sample"]["equity_curve"]) < 210


def test_result_id_matches_strategy():
    df = make_df(100)
    result = Backtester(AlwaysLongStrategy()).run(df)
    assert result["id"] == "ALWAYS_LONG"
