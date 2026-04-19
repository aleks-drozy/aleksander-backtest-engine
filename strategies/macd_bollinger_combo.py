import pandas as pd
from engine.base_strategy import Strategy


class MACDBollingerComboStrategy(Strategy):
    id = "MACD_BOLLINGER_COMBO"
    name = "MACD + Bollinger Combo"
    description = (
        "Dual-confirmation strategy: enters long when price is below the lower Bollinger Band "
        "AND MACD is bullish; short when price is above the upper band AND MACD is bearish. "
        "Exits when price reverts to the middle band or MACD flips."
    )
    direction = "both"
    params = {"bb_period": 20, "std_dev": 2, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]

        # Bollinger Bands
        mid = close.rolling(self.params["bb_period"]).mean()
        std = close.rolling(self.params["bb_period"]).std()
        upper = mid + self.params["std_dev"] * std
        lower = mid - self.params["std_dev"] * std

        # MACD
        fast_ema = close.ewm(span=self.params["macd_fast"], adjust=False).mean()
        slow_ema = close.ewm(span=self.params["macd_slow"], adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=self.params["macd_signal"], adjust=False).mean()
        macd_bullish = macd > signal_line

        position = 0.0
        signals = pd.Series(0.0, index=df.index)

        for i in range(len(df)):
            if pd.isna(mid.iloc[i]) or pd.isna(signal_line.iloc[i]):
                continue
            c = float(close.iloc[i])
            m = float(mid.iloc[i])
            u = float(upper.iloc[i])
            l = float(lower.iloc[i])
            bull = bool(macd_bullish.iloc[i])

            if position == 0.0:
                if c < l and bull:
                    position = 1.0   # BB oversold + MACD bullish confirmation
                elif c > u and not bull:
                    position = -1.0  # BB overbought + MACD bearish confirmation
            elif position == 1.0 and (c >= m or not bull):
                position = 0.0       # exit long — reverted to mid or MACD turned
            elif position == -1.0 and (c <= m or bull):
                position = 0.0       # exit short — reverted to mid or MACD turned

            signals.iloc[i] = position

        return signals
