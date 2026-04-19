import pandas as pd
from engine.base_strategy import Strategy


class SMACrossoverStrategy(Strategy):
    id = "SMA_CROSSOVER"
    name = "SMA Crossover"
    description = (
        "Golden cross on hourly NQ: long when 50-bar SMA crosses above 200-bar SMA; "
        "flat when it crosses below. Long-only to align with NQ's structural uptrend."
    )
    direction = "long_only"
    params = {"fast": 50, "slow": 200}
    # 2× ATR stop — enough room for NQ noise without holding through full reversals
    sl_atr_mult = 2.0
    tp_atr_mult = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = df["Close"].rolling(self.params["fast"]).mean()
        slow = df["Close"].rolling(self.params["slow"]).mean()

        signals = pd.Series(0.0, index=df.index)
        signals[fast > slow] = 1.0
        signals[fast < slow] = -1.0
        signals[fast.isna() | slow.isna()] = 0.0
        return signals
