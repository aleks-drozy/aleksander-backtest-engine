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

    def run(self, df: pd.DataFrame, n_wf_splits: int = 3) -> dict:
        """
        Walk-forward backtest with n_wf_splits expanding-IS folds.

        Structure (example, n=3):
          Fold 1: IS=[0 : K],         OOS=[K   : K+step]
          Fold 2: IS=[0 : K+step],    OOS=[K+step : K+2*step]
          Fold 3: IS=[0 : K+2*step],  OOS=[K+2*step : end]

        Where K = 60% of total bars and step = remaining bars / n_wf_splits.

        OOS metrics are averaged across all folds for robustness.
        OOS equity is chained (each fold starts from the previous fold's end value).
        The displayed IS period uses the final (largest) training set.
        """
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

        # --- Volatility-scaled position sizing ---
        # For strategies with real ATR stops (sl_mult < 9), scale each trade so
        # that a stop-out risks exactly 1% of capital.  Formula:
        #   size = risk_target / (sl_mult * ATR%)
        # Capped at 2× to prevent excessive leverage; floored at 0.1× to stay active.
        # ATR is lagged by 1 bar (same shift as entry) to avoid lookahead bias.
        # Strategies with sl_atr_mult >= 9 use the wide stop as a safety net only —
        # their natural exit is the signal, so keep unit sizing.
        if self.strategy.sl_atr_mult < 9.0:
            atr_pct = (atr / df["Close"].replace(0, float("nan"))).ffill().clip(lower=1e-6)
            raw_size = 0.01 / (self.strategy.sl_atr_mult * atr_pct)
            size = raw_size.clip(0.1, 2.0).shift(1).fillna(1.0)
            pos = pos * size

        pct_ret = df["Close"].pct_change().fillna(0)

        # Slippage and commissions deducted when position changes
        position_change = pos.diff().abs().fillna(0)
        slippage_cost = position_change * (SLIPPAGE_BPS / 10_000)
        commission_cost = position_change * COMMISSION_RATE

        net = pos * pct_ret - slippage_cost - commission_cost

        total_bars = len(df)

        # Compute bars_per_year from actual data so Sharpe scales correctly
        total_days = (df.index[-1] - df.index[0]).days
        years = total_days / 365.25
        bars_per_year = int(total_bars / years) if years > 0 else 252

        # --- Walk-forward splits ---
        initial_train = int(total_bars * 0.60)
        step = max(1, (total_bars - initial_train) // n_wf_splits)

        oos_metrics_all: list[dict] = []
        oos_equities: list[pd.Series] = []
        equity_carry = 10_000.0

        for fold_i in range(n_wf_splits):
            is_end = initial_train + fold_i * step
            oos_start = is_end
            oos_end = min(oos_start + step, total_bars)
            if oos_start >= total_bars:
                break

            oos_net_blk = net.iloc[oos_start:oos_end]
            oos_pos_blk = pos.iloc[oos_start:oos_end]
            oos_px_blk = df["Close"].iloc[oos_start:oos_end]

            # Chain equity: each fold starts from where the previous ended
            oos_eq_blk = (1 + oos_net_blk).cumprod() * equity_carry
            equity_carry = float(oos_eq_blk.iloc[-1])
            oos_equities.append(oos_eq_blk)

            fold_m = compute_metrics(oos_net_blk, oos_pos_blk, oos_px_blk, bars_per_year)
            oos_metrics_all.append(fold_m["metrics"])

        # Average scalar OOS metrics across folds; sum trade counts
        metric_keys = [k for k in oos_metrics_all[0] if k != "num_trades"]
        n_folds = len(oos_metrics_all)
        avg_oos_metrics = {
            k: round(sum(m[k] for m in oos_metrics_all) / n_folds, 2)
            for k in metric_keys
        }
        avg_oos_metrics["num_trades"] = int(sum(m["num_trades"] for m in oos_metrics_all))

        # IS: final (largest) training window — everything before the first OOS block
        final_is_end = initial_train + (n_folds - 1) * step
        is_net = net.iloc[:final_is_end]
        is_pos = pos.iloc[:final_is_end]
        is_px = df["Close"].iloc[:final_is_end]
        is_df_slice = df.iloc[:final_is_end]
        is_equity = (1 + is_net).cumprod() * 10_000

        # OOS: full chained equity and full date span
        oos_equity_full = pd.concat(oos_equities)
        oos_df_slice = df.iloc[initial_train:]

        return {
            "id": self.strategy.id,
            "name": self.strategy.name,
            "description": self.strategy.description,
            "direction": self.strategy.direction,
            "params": self.strategy.params,
            "in_sample": {
                "period": {
                    "start": is_df_slice.index[0].strftime("%Y-%m-%d"),
                    "end": is_df_slice.index[-1].strftime("%Y-%m-%d"),
                },
                **compute_metrics(is_net, is_pos, is_px, bars_per_year),
                "sampled": "weekly",
                "equity_curve": sample_equity_weekly(is_equity),
            },
            "out_of_sample": {
                "period": {
                    "start": oos_df_slice.index[0].strftime("%Y-%m-%d"),
                    "end": oos_df_slice.index[-1].strftime("%Y-%m-%d"),
                },
                "metrics": avg_oos_metrics,
                "sampled": "weekly",
                "equity_curve": sample_equity_weekly(oos_equity_full),
            },
        }
