import pandas as pd
from engine.base_strategy import Strategy
from engine.metrics import compute_metrics, sample_equity_weekly

SLIPPAGE_BPS = 2
COMMISSION_RATE = 0.0001  # ~$2 per trade as fraction of close price


class Backtester:
    def __init__(self, strategy: Strategy, train_pct: float = 0.70):
        self.strategy = strategy
        self.train_pct = train_pct

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range — adapts stop/target levels to current volatility."""
        high, low, close = df["High"], df["Low"], df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _apply_sl_tp(
        signals: pd.Series,
        prices: pd.Series,
        atr: pd.Series,
        sl_mult: float,
        tp_mult: float | None,
    ) -> pd.Series:
        """
        Enforce ATR-based stop-loss and optional take-profit on a signal series.
        Exits early when SL/TP is hit, overriding the strategy's natural exit bar.
        SL/TP distances are in price units: sl_mult × ATR and tp_mult × ATR.
        """
        pos = pd.Series(0.0, index=signals.index)
        entry_price: float | None = None
        entry_dir: float = 0.0

        for i in range(len(signals)):
            sig = float(signals.iloc[i])
            price = float(prices.iloc[i])
            a = float(atr.iloc[i])

            if pd.isna(a):
                pos.iloc[i] = entry_dir
                continue

            # Check SL / TP on open position
            if entry_dir != 0.0 and entry_price is not None:
                pnl_pts = (price - entry_price) * entry_dir  # positive = in profit

                if pnl_pts <= -(sl_mult * a):          # stop-loss hit
                    entry_dir = 0.0
                    entry_price = None
                    pos.iloc[i] = 0.0
                    continue

                if tp_mult is not None and pnl_pts >= tp_mult * a:  # take-profit hit
                    entry_dir = 0.0
                    entry_price = None
                    pos.iloc[i] = 0.0
                    continue

            # Process signal
            if entry_dir == 0.0:
                if sig != 0.0:                  # new entry
                    entry_dir = sig
                    entry_price = price
            elif sig == 0.0:                    # strategy says exit
                entry_dir = 0.0
                entry_price = None
            elif sig != entry_dir:              # direction flip
                entry_dir = sig
                entry_price = price

            pos.iloc[i] = entry_dir

        return pos

    def run(self, df: pd.DataFrame) -> dict:
        signals = self.strategy.generate_signals(df)

        if self.strategy.direction == "long_only":
            signals = signals.clip(lower=0)
        elif self.strategy.direction == "short_only":
            signals = signals.clip(upper=0)

        # Apply ATR-based stop-loss / take-profit before shifting into positions
        atr = self._compute_atr(df, self.strategy.atr_period)
        signals = self._apply_sl_tp(
            signals, df["Close"], atr,
            self.strategy.sl_atr_mult,
            self.strategy.tp_atr_mult,
        )

        # Enter position on next bar to avoid lookahead bias
        pos = signals.shift(1).fillna(0)

        pct_ret = df["Close"].pct_change().fillna(0)

        # Slippage and commissions deducted when position changes
        position_change = pos.diff().abs().fillna(0)
        slippage_cost = position_change * (SLIPPAGE_BPS / 10_000)
        commission_cost = position_change * COMMISSION_RATE

        net = pos * pct_ret - slippage_cost - commission_cost

        split_idx = int(len(df) * self.train_pct)

        # Compute bars_per_year from actual data so Sharpe scales correctly
        # for any timeframe (daily, hourly, etc.)
        total_days = (df.index[-1] - df.index[0]).days
        years = total_days / 365.25
        bars_per_year = int(len(df) / years) if years > 0 else 252

        is_net = net.iloc[:split_idx]
        oos_net = net.iloc[split_idx:]
        is_pos = pos.iloc[:split_idx]
        oos_pos = pos.iloc[split_idx:]
        is_prices = df["Close"].iloc[:split_idx]
        oos_prices = df["Close"].iloc[split_idx:]
        is_df = df.iloc[:split_idx]
        oos_df = df.iloc[split_idx:]

        is_equity = (1 + is_net).cumprod() * 10_000
        oos_equity = (1 + oos_net).cumprod() * 10_000

        return {
            "id": self.strategy.id,
            "name": self.strategy.name,
            "description": self.strategy.description,
            "direction": self.strategy.direction,
            "params": self.strategy.params,
            "in_sample": {
                "period": {
                    "start": is_df.index[0].strftime("%Y-%m-%d"),
                    "end": is_df.index[-1].strftime("%Y-%m-%d"),
                },
                **compute_metrics(is_net, is_pos, is_prices, bars_per_year),
                "sampled": "weekly",
                "equity_curve": sample_equity_weekly(is_equity),
            },
            "out_of_sample": {
                "period": {
                    "start": oos_df.index[0].strftime("%Y-%m-%d"),
                    "end": oos_df.index[-1].strftime("%Y-%m-%d"),
                },
                **compute_metrics(oos_net, oos_pos, oos_prices, bars_per_year),
                "sampled": "weekly",
                "equity_curve": sample_equity_weekly(oos_equity),
            },
        }
