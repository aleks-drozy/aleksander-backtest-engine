import pandas as pd
from engine.base_strategy import Strategy


class OpeningRangeBreakoutStrategy(Strategy):
    """
    Opening Range Breakout (ORB) on 1-hour bars, US cash session (09:30-16:00 ET).

    Opening Range: high and low of the very first bar each day (09:30-10:30).
    Entry: go long on the first bar that closes ABOVE the OR high;
           go short on the first bar that closes BELOW the OR low.
    One trade per day; position is held until session close (last bar set to 0).
    ATR stop-loss applied by the backtester to limit runaway losers.

    Directional edge: intraday momentum tends to continue in the direction of the
    opening break (established by the first hour's range).
    """

    id = "ORB"
    name = "Opening Range Breakout"
    description = (
        "Trades the breakout above/below the first 1h bar's high/low each session. "
        "One trade per day; held to close. Captures intraday momentum on NQ futures."
    )
    direction = "both"
    params = {"or_bars": 1}

    # 2x ATR stop to protect against gap reversals; let momentum run to session close
    sl_atr_mult: float = 2.0
    tp_atr_mult: float | None = None
    atr_period: int = 14

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0.0, index=df.index)

        # Group bars by calendar date
        dates = df.index.normalize()
        unique_dates = dates.unique()

        for date in unique_dates:
            day_idx = df.index[dates == date]

            # Need at least the OR bar + 1 signal bar
            if len(day_idx) < 2:
                continue

            # Opening range: high/low of the first bar
            or_high = float(df.loc[day_idx[0], "High"])
            or_low = float(df.loc[day_idx[0], "Low"])

            trade_dir = 0.0  # track direction taken this session

            # Signal bars start from bar index 1 (after the OR bar)
            for bar_idx in day_idx[1:]:
                if trade_dir == 0.0:
                    close = float(df.loc[bar_idx, "Close"])
                    if close > or_high:
                        trade_dir = 1.0   # breakout long
                    elif close < or_low:
                        trade_dir = -1.0  # breakdown short
                signals.loc[bar_idx] = trade_dir

            # Force flat on the last bar of the day so position
            # closes at the session-close bar's return
            signals.loc[day_idx[-1]] = 0.0

        return signals
