from abc import ABC, abstractmethod
from typing import Literal
import pandas as pd


class Strategy(ABC):
    id: str
    name: str
    description: str
    direction: Literal["long_only", "short_only", "both"]
    params: dict

    # ATR-based risk management — override per strategy
    # sl_atr_mult: stop distance in ATR multiples (always set)
    # tp_atr_mult: take-profit distance in ATR multiples (None = use signal exit only)
    # atr_period:  lookback for ATR calculation
    sl_atr_mult: float = 2.0
    tp_atr_mult: float | None = None
    atr_period: int = 14

    @staticmethod
    def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average Directional Index — measures trend strength (not direction).
        ADX > 25: trending market (favour trend-following signals)
        ADX < 25: ranging market (favour mean-reversion signals)
        """
        high, low, close = df["High"], df["Low"], df["Close"]
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        plus_dm = (high - prev_high).clip(lower=0)
        minus_dm = (prev_low - low).clip(lower=0)
        # Zero out whichever DM is not dominant
        mask = plus_dm >= minus_dm
        plus_dm = plus_dm.where(mask, 0.0)
        minus_dm = minus_dm.where(~mask, 0.0)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of 1 (long), -1 (short), 0 (flat) aligned to df.index."""
        ...
