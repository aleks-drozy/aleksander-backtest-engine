import pandas as pd
from engine.base_strategy import Strategy


class BollingerMeanReversionStrategy(Strategy):
    id = "BOLLINGER_MEAN_REVERSION"
    name = "Bollinger Mean Reversion"
    description = (
        "Long when price dips below the lower Bollinger Band; short when it spikes above "
        "the upper band. Exits when price reverts to the middle band (20-period SMA)."
    )
    direction = "both"
    params = {"period": 20, "std_dev": 2}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        mid = close.rolling(self.params["period"]).mean()
        std = close.rolling(self.params["period"]).std()
        upper = mid + self.params["std_dev"] * std
        lower = mid - self.params["std_dev"] * std

        position = 0.0
        signals = pd.Series(0.0, index=df.index)

        for i in range(len(df)):
            if pd.isna(mid.iloc[i]):
                continue
            c = float(close.iloc[i])
            m = float(mid.iloc[i])
            u = float(upper.iloc[i])
            l = float(lower.iloc[i])

            if position == 0.0:
                if c < l:
                    position = 1.0   # enter long — price below lower band
                elif c > u:
                    position = -1.0  # enter short — price above upper band
            elif position == 1.0 and c >= m:
                position = 0.0       # exit long at middle band
            elif position == -1.0 and c <= m:
                position = 0.0       # exit short at middle band

            signals.iloc[i] = position

        return signals
