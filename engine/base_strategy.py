from abc import ABC, abstractmethod
from typing import Literal
import pandas as pd


class Strategy(ABC):
    id: str
    name: str
    description: str
    direction: Literal["long_only", "short_only", "both"]
    params: dict

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of 1 (long), -1 (short), 0 (flat) aligned to df.index."""
        ...
