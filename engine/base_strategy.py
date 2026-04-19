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

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of 1 (long), -1 (short), 0 (flat) aligned to df.index."""
        ...
