from typing import Literal
from pydantic import BaseModel


class EquityPoint(BaseModel):
    date: str
    value: float


class Metrics(BaseModel):
    total_return_pct: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    num_trades: int
    # Statistical validation
    probabilistic_sharpe: float   # P(true SR > 0) corrected for non-normality; >0.95 = significant
    monte_carlo_p_value: float    # Fraction of permutations beating actual SR; <0.05 = significant


class Period(BaseModel):
    start: str
    end: str


class PeriodResult(BaseModel):
    period: Period
    metrics: Metrics
    sampled: Literal["weekly"]
    equity_curve: list[EquityPoint]


class ParamSensitivityEntry(BaseModel):
    param: str
    values: list[float]
    oos_sharpes: list[float]


class StrategyResult(BaseModel):
    id: str
    name: str
    description: str
    direction: Literal["long_only", "short_only", "both"]
    timeframe: str
    params: dict[str, float | int]
    in_sample: PeriodResult
    out_of_sample: PeriodResult
    # Risk / robustness diagnostics
    kelly_fraction: float                         # half-Kelly optimal leverage (>1 = strategy warrants leverage)
    cost_sensitivity: dict[str, float]            # OOS Sharpe at {"1x": ..., "2x": ..., "4x": ...} cost levels
    param_sensitivity: list[ParamSensitivityEntry]  # OOS Sharpe across ±20% param grid


class BacktestResults(BaseModel):
    generated_at: str
    asset_universe: dict[str, str]
    benchmark_return_pct: float   # NQ buy-and-hold over the OOS window
    strategies: list[StrategyResult]
    correlation_matrix: dict[str, dict[str, float]]  # pairwise daily P&L correlation between strategies
