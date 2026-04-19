import pandas as pd
from engine.base_strategy import Strategy


class MACDCrossoverStrategy(Strategy):
    id = "MACD_CROSSOVER"
    name = "MACD Crossover"
    description = (
        "Long when MACD line crosses above the signal line; short when it crosses below. "
        "Uses exponential moving averages (12/26/9) to capture momentum shifts."
    )
    direction = "both"
    params = {"fast": 12, "slow": 26, "signal": 9}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        fast_ema = close.ewm(span=self.params["fast"], adjust=False).mean()
        slow_ema = close.ewm(span=self.params["slow"], adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=self.params["signal"], adjust=False).mean()

        signals = pd.Series(0.0, index=df.index)
        signals[macd > signal_line] = 1.0
        signals[macd < signal_line] = -1.0

        # Zero out early bars before indicators converge
        warmup = self.params["slow"] + self.params["signal"]
        signals.iloc[:warmup] = 0.0
        return signals
