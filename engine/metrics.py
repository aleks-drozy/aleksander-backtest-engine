import numpy as np
import pandas as pd

def compute_metrics(
    net_returns: pd.Series,
    positions: pd.Series,
    prices: pd.Series,
    bars_per_year: int = 252,
) -> dict:
    equity = (1 + net_returns).cumprod()

    total_return_pct = round((equity.iloc[-1] - 1) * 100, 2)

    sharpe = 0.0
    if net_returns.std() > 0:
        sharpe = round(
            (net_returns.mean() / net_returns.std()) * np.sqrt(bars_per_year), 2
        )

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown_pct = round(float(drawdown.min()) * 100, 2)

    trades = _extract_trades(positions, prices)
    num_trades = len(trades)
    win_rate_pct = 0.0
    profit_factor = 0.0

    if num_trades > 0:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        win_rate_pct = round(len(wins) / num_trades * 100, 2)
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0

    return {
        "metrics": {
            "total_return_pct": total_return_pct,
            "sharpe": sharpe,
            "max_drawdown_pct": max_drawdown_pct,
            "win_rate_pct": win_rate_pct,
            "profit_factor": profit_factor,
            "num_trades": num_trades,
        }
    }


def sample_equity_weekly(equity: pd.Series) -> list[dict]:
    weekly = equity.resample("W").last().dropna()
    return [
        {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
        for d, v in weekly.items()
    ]


def _extract_trades(positions: pd.Series, prices: pd.Series) -> list[float]:
    trades: list[float] = []
    entry_price: float | None = None
    entry_dir: float = 0.0

    for i in range(len(positions)):
        p = float(positions.iloc[i])
        price = float(prices.iloc[i])

        if entry_dir == 0.0 and p != 0.0:
            entry_price = price
            entry_dir = p
        elif entry_dir != 0.0 and (p == 0.0 or p != entry_dir):
            assert entry_price is not None
            pnl = (price / entry_price - 1) * entry_dir
            trades.append(pnl)
            entry_dir = 0.0
            entry_price = None
            if p != 0.0:
                entry_price = price
                entry_dir = p

    return trades
