import pandas as pd
from engine.base_strategy import Strategy
from engine.metrics import compute_metrics, sample_equity_weekly

SLIPPAGE_BPS = 2
COMMISSION_RATE = 0.0001  # ~$2 per trade as fraction of close price


class Backtester:
    def __init__(self, strategy: Strategy, train_pct: float = 0.70):
        self.strategy = strategy
        self.train_pct = train_pct

    def run(self, df: pd.DataFrame) -> dict:
        signals = self.strategy.generate_signals(df)

        if self.strategy.direction == "long_only":
            signals = signals.clip(lower=0)
        elif self.strategy.direction == "short_only":
            signals = signals.clip(upper=0)

        # Enter position on next bar to avoid lookahead bias
        pos = signals.shift(1).fillna(0)

        pct_ret = df["Close"].pct_change().fillna(0)

        # Slippage and commissions deducted when position changes
        position_change = pos.diff().abs().fillna(0)
        slippage_cost = position_change * (SLIPPAGE_BPS / 10_000)
        commission_cost = position_change * COMMISSION_RATE

        net = pos * pct_ret - slippage_cost - commission_cost

        split_idx = int(len(df) * self.train_pct)

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
                **compute_metrics(is_net, is_pos, is_prices),
                "sampled": "weekly",
                "equity_curve": sample_equity_weekly(is_equity),
            },
            "out_of_sample": {
                "period": {
                    "start": oos_df.index[0].strftime("%Y-%m-%d"),
                    "end": oos_df.index[-1].strftime("%Y-%m-%d"),
                },
                **compute_metrics(oos_net, oos_pos, oos_prices),
                "sampled": "weekly",
                "equity_curve": sample_equity_weekly(oos_equity),
            },
        }
