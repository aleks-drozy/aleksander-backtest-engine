import pandas as pd
from engine.base_strategy import Strategy


class IFVGCISDStrategy(Strategy):
    """Daily-timeframe adaptation of the FYP IFVG+CISD strategy.

    IFVG: detects 3-candle imbalances and fires when price closes back through
    the gap level, inverting the prior bias.

    CISD: simplified as a rolling N-bar breakout — a close above the prior
    N-bar high (bullish structural shift) or below the prior N-bar low.

    Both signals must agree in the same direction for an entry.
    """

    id = "IFVG_CISD"
    name = "IFVG + CISD"
    description = (
        "Hourly adaptation of the FYP strategy: requires both a bullish Inverse Fair Value Gap "
        "and a Change in State of Delivery (breakout above prior 20-bar high) to confirm. "
        "Long-only to align with NQ's structural uptrend."
    )
    direction = "long_only"
    params = {"ifvg_lookback": 5, "cisd_window": 20}
    # 1:2 R:R — win rate ~56% needs losers to be half the size of winners
    sl_atr_mult = 1.0
    tp_atr_mult = 2.0

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lb = self.params["ifvg_lookback"]
        cw = self.params["cisd_window"]

        # Bearish FVG: current high < low from 2 bars ago (gap down)
        bearish_fvg_mask = df["High"] < df["Low"].shift(2)
        bearish_fvg_level = df["Low"].shift(2).where(bearish_fvg_mask)

        # Bullish FVG: current low > high from 2 bars ago (gap up)
        bullish_fvg_mask = df["Low"] > df["High"].shift(2)
        bullish_fvg_level = df["High"].shift(2).where(bullish_fvg_mask)

        # Bullish IFVG: close above a prior bearish FVG level
        bearish_ref = bearish_fvg_level.shift(1).ffill(limit=lb)
        bullish_ifvg = df["Close"] > bearish_ref

        # Bearish IFVG: close below a prior bullish FVG level
        bullish_ref = bullish_fvg_level.shift(1).ffill(limit=lb)
        bearish_ifvg = df["Close"] < bullish_ref

        # CISD: rolling N-bar breakout
        prior_high = df["High"].rolling(cw).max().shift(1)
        prior_low = df["Low"].rolling(cw).min().shift(1)
        bullish_cisd = df["Close"] > prior_high
        bearish_cisd = df["Close"] < prior_low

        signals = pd.Series(0.0, index=df.index)
        signals[bullish_ifvg & bullish_cisd] = 1.0
        signals[bearish_ifvg & bearish_cisd] = -1.0
        return signals
