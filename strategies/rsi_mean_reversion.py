import numpy as np
import pandas as pd
from engine.base_strategy import Strategy


class RSIMeanReversionStrategy(Strategy):
    id = "RSI_MEAN_REVERSION"
    name = "RSI Mean Reversion"
    description = (
        "Long when RSI dips below 35; short when RSI spikes above 65. "
        "Exits when RSI reverts to the opposite threshold. "
        "Wider thresholds than classic 30/70 to generate meaningful trade frequency on 1h NQ."
    )
    direction = "both"
    params = {"period": 14, "oversold": 35, "overbought": 65}
    # Natural RSI threshold crossing IS the exit — stop only for catastrophic tail events
    sl_atr_mult = 10.0
    tp_atr_mult = None

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = self._rsi(df["Close"], self.params["period"])

        position = 0.0  # 0 = flat, 1 = long, -1 = short
        signals = pd.Series(0.0, index=df.index)

        for i in range(len(df)):
            if pd.isna(rsi.iloc[i]):
                continue
            r = rsi.iloc[i]
            if position == 0.0:
                if r < self.params["oversold"]:
                    position = 1.0   # enter long on oversold dip
                elif r > self.params["overbought"]:
                    position = -1.0  # enter short on overbought spike
            elif position == 1.0 and r > self.params["overbought"]:
                position = 0.0       # exit long when overbought
            elif position == -1.0 and r < self.params["oversold"]:
                position = 0.0       # exit short when oversold
            signals.iloc[i] = position

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
