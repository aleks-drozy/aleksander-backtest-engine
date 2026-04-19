import pandas as pd
from engine.base_strategy import Strategy


class MACDCrossoverStrategy(Strategy):
    id = "MACD_CROSSOVER"
    name = "MACD Crossover"
    description = (
        "Long when MACD line crosses above the signal line; flat when it crosses below. "
        "Uses 48/104/18 EMAs on 1h bars: fast ~2 days, slow ~2 weeks. "
        "Long-only to align with NQ's structural upward bias."
    )
    direction = "long_only"
    params = {"fast": 48, "slow": 104, "signal": 18}
    # 1:3 R:R — 2.5× ATR stop gives trades room; target is 3× ATR on confirmed moves
    sl_atr_mult = 2.5
    tp_atr_mult = 3.0

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
