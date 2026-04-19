import pytest
from pydantic import ValidationError
from schemas import BacktestResults, StrategyResult, PeriodResult, Metrics, EquityPoint


def _valid_period() -> dict:
    return {
        "period": {"start": "2020-01-01", "end": "2022-12-31"},
        "metrics": {
            "total_return_pct": 25.4,
            "sharpe": 1.1,
            "max_drawdown_pct": -12.3,
            "win_rate_pct": 55.0,
            "profit_factor": 1.4,
            "num_trades": 30,
        },
        "sampled": "weekly",
        "equity_curve": [{"date": "2020-01-05", "value": 10000.0}],
    }


def _valid_strategy() -> dict:
    return {
        "id": "SMA_CROSSOVER",
        "name": "SMA Crossover",
        "description": "Goes long on golden cross.",
        "direction": "long_only",
        "timeframe": "1h",
        "params": {"fast": 20, "slow": 50},
        "in_sample": _valid_period(),
        "out_of_sample": _valid_period(),
    }


def test_valid_backtest_results_parses():
    data = {
        "generated_at": "2026-04-18T12:00:00Z",
        "asset_universe": {"SMA_CROSSOVER": "SPY"},
        "strategies": [_valid_strategy()],
    }
    result = BacktestResults(**data)
    assert result.strategies[0].id == "SMA_CROSSOVER"


def test_invalid_direction_raises():
    s = _valid_strategy()
    s["direction"] = "sideways"
    with pytest.raises(ValidationError):
        StrategyResult(**s)


def test_missing_metric_key_raises():
    p = _valid_period()
    del p["metrics"]["sharpe"]
    with pytest.raises(ValidationError):
        PeriodResult(**p)


def test_invalid_sampled_value_raises():
    p = _valid_period()
    p["sampled"] = "daily"
    with pytest.raises(ValidationError):
        PeriodResult(**p)
