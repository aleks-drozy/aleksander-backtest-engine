from typing import Literal
from pydantic import BaseModel


class EquityPoint(BaseModel):
    date: str
    value: float


class Metrics(BaseModel):
    total_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    num_trades: int


class Period(BaseModel):
    start: str
    end: str


class PeriodResult(BaseModel):
    period: Period
    metrics: Metrics
    sampled: Literal["weekly"]
    equity_curve: list[EquityPoint]


class StrategyResult(BaseModel):
    id: str
    name: str
    description: str
    direction: Literal["long_only", "short_only", "both"]
    timeframe: str
    params: dict[str, float | int]
    in_sample: PeriodResult
    out_of_sample: PeriodResult


class BacktestResults(BaseModel):
    generated_at: str
    asset_universe: dict[str, str]
    strategies: list[StrategyResult]
