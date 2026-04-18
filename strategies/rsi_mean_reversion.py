import numpy as np
import pandas as pd
from engine.base_strategy import Strategy


class RSIMeanReversionStrategy(Strategy):
    id = "RSI_MEAN_REVERSION"
    name = "RSI Mean Reversion"
    description = (
        "Enters long when RSI crosses above the oversold threshold; "
        "exits when RSI crosses above the overbought threshold."
    )
    direction = "long_only"
    params = {"period": 14, "oversold": 30, "overbought": 70}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = self._rsi(df["Close"], self.params["period"])

        in_position = False
        signals = pd.Series(0.0, index=df.index)

        for i in range(len(df)):
            if pd.isna(rsi.iloc[i]):
                continue
            if not in_position and rsi.iloc[i] < self.params["oversold"]:
                in_position = True
            elif in_position and rsi.iloc[i] > self.params["overbought"]:
                in_position = False
            signals.iloc[i] = 1.0 if in_position else 0.0

        return signals

    @staticmethod
    def _rsi(prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
