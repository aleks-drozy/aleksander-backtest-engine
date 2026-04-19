import pandas as pd
from engine.base_strategy import Strategy


class SMACrossoverStrategy(Strategy):
    id = "SMA_CROSSOVER"
    name = "SMA Crossover"
    description = (
        "Goes long when fast SMA crosses above slow SMA; short when it crosses below."
    )
    direction = "both"
    params = {"fast": 20, "slow": 50}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = df["Close"].rolling(self.params["fast"]).mean()
        slow = df["Close"].rolling(self.params["slow"]).mean()

        signals = pd.Series(0.0, index=df.index)
        signals[fast > slow] = 1.0
        signals[fast < slow] = -1.0
        signals[fast.isna() | slow.isna()] = 0.0
        return signals
