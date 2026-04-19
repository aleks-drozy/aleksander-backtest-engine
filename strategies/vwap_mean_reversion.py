import pandas as pd
from engine.base_strategy import Strategy


class VWAPMeanReversionStrategy(Strategy):
    """
    VWAP Mean Reversion on 1-hour bars, US cash session (09:30-16:00 ET).

    VWAP (Volume-Weighted Average Price) is the most important intraday price
    level for institutional NQ flow.  When price deviates significantly from the
    session VWAP, algorithmic market-makers and institutional desks rebalance
    back toward it — creating a predictable mean-reversion tendency.

    Entry:
        Long  when Close < VWAP × (1 - threshold_pct)   [price below VWAP]
        Short when Close > VWAP × (1 + threshold_pct)   [price above VWAP]

    Exit:
        Price returns within 0.1% of VWAP, OR max_hold_bars elapsed.
        Position always closed on the session's last bar.

    VWAP resets to zero at the start of each session.
    No ADX filter needed — VWAP edge is regime-agnostic (works in trends too,
    because even trend days oscillate around VWAP intraday).
    """

    id = "VWAP_MEAN_REVERSION"
    name = "VWAP Mean Reversion"
    description = (
        "Fades price deviations >0.6% from the session VWAP. "
        "VWAP resets daily; exits when price reverts or after 4 bars max. "
        "Exploits institutional rebalancing flow on NQ 1h."
    )
    direction = "both"
    params = {"threshold_pct": 0.006, "max_hold_bars": 4}

    # ATR stop for gap/spike protection; VWAP reversion is the primary exit
    sl_atr_mult: float = 2.0
    tp_atr_mult: float | None = None
    atr_period: int = 14

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0.0, index=df.index)

        dates = df.index.normalize()
        unique_dates = dates.unique()

        for date in unique_dates:
            day_idx = df.index[dates == date]
            if len(day_idx) < 3:
                continue

            day_df = df.loc[day_idx]

            # Intraday VWAP: cumulative (typical_price × volume) / cumulative volume
            typical_price = (day_df["High"] + day_df["Low"] + day_df["Close"]) / 3
            vol = day_df["Volume"].fillna(0).clip(lower=1.0)  # avoid div-by-zero
            cum_tp_vol = (typical_price * vol).cumsum()
            cum_vol = vol.cumsum()
            vwap = cum_tp_vol / cum_vol

            threshold = self.params["threshold_pct"]
            max_hold = int(self.params["max_hold_bars"])

            position = 0.0
            hold_bars = 0

            for bar_idx in day_idx:
                v = float(vwap.loc[bar_idx])
                c = float(df.loc[bar_idx, "Close"])

                if position != 0.0:
                    hold_bars += 1
                    dev_from_vwap = abs(c - v) / v
                    if dev_from_vwap < 0.001 or hold_bars >= max_hold:
                        position = 0.0  # reverted to VWAP or max hold

                if position == 0.0:
                    dev = (c - v) / v
                    if dev < -threshold:
                        position = 1.0   # long: below VWAP
                        hold_bars = 0
                    elif dev > threshold:
                        position = -1.0  # short: above VWAP
                        hold_bars = 0

                signals.loc[bar_idx] = position

            # Always flat at session close
            signals.loc[day_idx[-1]] = 0.0

        return signals
