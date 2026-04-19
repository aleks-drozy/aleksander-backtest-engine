import pytest
from pydantic import ValidationError
from schemas import BacktestResults, StrategyResult, PeriodResult, Metrics, EquityPoint


def _valid_metrics() -> dict:
    return {
        "total_return_pct": 25.4,
        "sharpe": 1.1,
        "sortino": 1.6,
        "calmar": 2.1,
        "max_drawdown_pct": -12.3,
        "win_rate_pct": 55.0,
        "profit_factor": 1.4,
        "num_trades": 30,
        "probabilistic_sharpe": 0.92,
        "monte_carlo_p_value": 0.03,
    }


def _valid_period() -> dict:
    return {
        "period": {"start": "2020-01-01", "end": "2022-12-31"},
        "metrics": _valid_metrics(),
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
        "kelly_fraction": 0.85,
        "cost_sensitivity": {"1x": 1.1, "2x": 0.7, "4x": 0.1},
        "param_sensitivity": [
            {"param": "fast", "values": [16.0, 18.0, 20.0, 22.0, 24.0], "oos_sharpes": [0.9, 1.0, 1.1, 1.0, 0.8]},
        ],
    }


def test_valid_backtest_results_parses():
    data = {
        "generated_at": "2026-04-18T12:00:00Z",
        "asset_universe": {"SMA_CROSSOVER": "SPY"},
        "benchmark_return_pct": 12.5,
        "strategies": [_valid_strategy()],
        "correlation_matrix": {"SMA_CROSSOVER": {"SMA_CROSSOVER": 1.0}},
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


def test_probabilistic_sharpe_field_present():
    m = Metrics(**_valid_metrics())
    assert hasattr(m, "probabilistic_sharpe")
    assert 0.0 <= m.probabilistic_sharpe <= 1.0


def test_cost_sensitivity_stored():
    s = StrategyResult(**_valid_strategy())
    assert "1x" in s.cost_sensitivity
    assert "2x" in s.cost_sensitivity
    assert "4x" in s.cost_sensitivity


def test_param_sensitivity_stored():
    s = StrategyResult(**_valid_strategy())
    assert len(s.param_sensitivity) >= 1
    entry = s.param_sensitivity[0]
    assert len(entry.values) == len(entry.oos_sharpes)


def test_correlation_matrix_stored():
    data = {
        "generated_at": "2026-04-18T12:00:00Z",
        "asset_universe": {"SMA_CROSSOVER": "SPY"},
        "benchmark_return_pct": 12.5,
        "strategies": [_valid_strategy()],
        "correlation_matrix": {"SMA_CROSSOVER": {"SMA_CROSSOVER": 1.0}},
    }
    result = BacktestResults(**data)
    assert result.correlation_matrix["SMA_CROSSOVER"]["SMA_CROSSOVER"] == 1.0
