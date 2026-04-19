import pandas as pd
from engine.base_strategy import Strategy


class OpeningRangeBreakoutStrategy(Strategy):
    """
    Volume-Confirmed Opening Range Breakout (ORB) on 1-hour bars.

    Opening Range: high and low of the very first bar each day (09:30-10:30).

    Entry: first bar that closes ABOVE the OR high AND whose volume exceeds
           vol_mult × the average volume of the OR bar(s).  Confirmed breakouts
           have institutional participation; unconfirmed ones are fakeouts ~50%.

    Short entry mirrors the long: close below OR low with volume confirmation.

    One trade per day; position held to session close (last bar forced to 0).
    ATR stop-loss applied by backtester for runaway-gap protection.
    """

    id = "ORB"
    name = "Opening Range Breakout"
    description = (
        "Trades volume-confirmed breakouts above/below the first 1h bar each session. "
        "Volume must exceed 1.5x OR-bar average to confirm institutional participation. "
        "One trade per day; held to close. Captures intraday momentum on NQ futures."
    )
    direction = "both"
    params = {"or_bars": 1, "vol_mult": 1.5}

    sl_atr_mult: float = 2.0
    tp_atr_mult: float | None = None
    atr_period: int = 14

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0.0, index=df.index)
        has_volume = "Volume" in df.columns and df["Volume"].notna().any()

        dates = df.index.normalize()
        unique_dates = dates.unique()

        for date in unique_dates:
            day_idx = df.index[dates == date]
            if len(day_idx) < 2:
                continue

            or_high = float(df.loc[day_idx[0], "High"])
            or_low  = float(df.loc[day_idx[0], "Low"])

            # Average volume of the opening range bar(s) — used for confirmation
            or_vol_avg = float(df.loc[day_idx[0], "Volume"]) if has_volume else 0.0

            trade_dir = 0.0

            for bar_idx in day_idx[1:]:
                if trade_dir == 0.0:
                    close = float(df.loc[bar_idx, "Close"])

                    # Volume confirmation: breakout bar must have elevated volume
                    if has_volume and or_vol_avg > 0:
                        bar_vol = float(df.loc[bar_idx, "Volume"])
                        vol_ok = bar_vol >= self.params["vol_mult"] * or_vol_avg
                    else:
                        vol_ok = True  # no volume data → skip confirmation

                    if close > or_high and vol_ok:
                        trade_dir = 1.0
                    elif close < or_low and vol_ok:
                        trade_dir = -1.0

                signals.loc[bar_idx] = trade_dir

            # Force flat at session close
            signals.loc[day_idx[-1]] = 0.0

        return signals
