import pandas as pd
from engine.base_strategy import Strategy


class OvernightGapFadeStrategy(Strategy):
    """
    Overnight Gap Fade on 1-hour bars, US cash session (09:30-16:00 ET).

    NQ futures open with a gap vs the prior session's close roughly 60% of
    sessions.  Small gaps (0.1-0.3%) fill within the first 1-2 hours ~65-70%
    of the time historically — driven by retail over-reaction at open that
    institutions fade back toward fair value.  Large gaps (>0.5%) are
    continuation moves; those are NOT traded here.

    Entry:
        At the first bar of each session (09:30 open), compare the open to the
        prior session's last close.  If the gap is within [min_gap_pct,
        max_gap_pct], fade it: short an up-gap, long a down-gap.

    Exit:
        Gap fills (price crosses back through prior close), OR max_hold_bars
        elapsed, OR session ends.  Position always closed on session's last bar.
    """

    id = "OVERNIGHT_GAP_FADE"
    name = "Overnight Gap Fade"
    description = (
        "Fades small overnight gaps (0.1-0.3%) on NQ open. "
        "Enters on the 09:30 bar; exits on gap fill or after 2 bars. "
        "Exploits mean-reversion of retail over-reaction at the open."
    )
    direction = "both"
    params = {"min_gap_pct": 0.001, "max_gap_pct": 0.003, "max_hold_bars": 2}

    # Tighter stop: gap fades either work fast or they don't
    sl_atr_mult: float = 1.5
    tp_atr_mult: float | None = None
    atr_period: int = 14

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0.0, index=df.index)

        dates = df.index.normalize()
        unique_dates = sorted(dates.unique())

        prev_close: float | None = None

        for date in unique_dates:
            day_idx = df.index[dates == date]
            if len(day_idx) == 0:
                continue

            first_bar = day_idx[0]
            open_price = float(df.loc[first_bar, "Open"])

            if prev_close is not None and prev_close > 0:
                gap_pct = (open_price - prev_close) / prev_close
                abs_gap = abs(gap_pct)
                min_g = self.params["min_gap_pct"]
                max_g = self.params["max_gap_pct"]

                if min_g <= abs_gap <= max_g:
                    # Fade the gap: direction is OPPOSITE to the gap direction
                    fade_dir = -1.0 if gap_pct > 0 else 1.0
                    target = prev_close   # gap is "filled" when price reaches prior close
                    position = fade_dir
                    hold_bars = 0

                    for bar_idx in day_idx:
                        c = float(df.loc[bar_idx, "Close"])
                        hold_bars += 1

                        # Check fill: long target = price >= prior close; short = price <= prior close
                        filled = (fade_dir > 0 and c >= target) or (fade_dir < 0 and c <= target)
                        if filled or hold_bars > self.params["max_hold_bars"]:
                            position = 0.0

                        signals.loc[bar_idx] = position
                        if position == 0.0:
                            break  # stop iterating once flat

            # Record last bar's close as prior close for next session
            prev_close = float(df.loc[day_idx[-1], "Close"])

        return signals
