import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_metrics(
    net_returns: pd.Series,
    positions: pd.Series,
    prices: pd.Series,
    bars_per_year: int = 252,
) -> dict:
    equity = (1 + net_returns).cumprod()

    total_return_pct = round((equity.iloc[-1] - 1) * 100, 2)

    # Sharpe — penalises all volatility (up and down)
    sharpe = 0.0
    ret_std = net_returns.std()
    ret_mean = net_returns.mean()
    if ret_std > 0:
        sharpe = round((ret_mean / ret_std) * np.sqrt(bars_per_year), 2)

    # Sortino — penalises only downside volatility; more relevant for trading
    sortino = 0.0
    downside = net_returns[net_returns < 0]
    downside_std = downside.std()
    if downside_std > 0:
        sortino = round((ret_mean / downside_std) * np.sqrt(bars_per_year), 2)

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown_pct = round(float(drawdown.min()) * 100, 2)

    # Calmar — annualised return divided by max drawdown (how hard the strategy works)
    calmar = 0.0
    if max_drawdown_pct < 0:
        annualised_return = (equity.iloc[-1] ** (bars_per_year / len(net_returns)) - 1) * 100
        calmar = round(annualised_return / abs(max_drawdown_pct), 2)

    # Probabilistic Sharpe Ratio (Bailey & López de Prado, 2014)
    # PSR = P(true SR > 0 | observed SR) adjusted for non-normality of returns
    probabilistic_sharpe = 0.5
    n_obs = len(net_returns)
    if n_obs > 4 and ret_std > 0:
        sr_per_bar = ret_mean / ret_std          # un-annualised SR
        skew = float(net_returns.skew())
        # scipy/pandas kurtosis() returns EXCESS kurtosis (0 for normal);
        # formula needs standard kurtosis (3 for normal)
        std_kurt = float(net_returns.kurtosis()) + 3.0
        denom_sq = 1.0 - skew * sr_per_bar + (std_kurt - 1.0) / 4.0 * sr_per_bar ** 2
        if denom_sq > 0:
            z = (sr_per_bar * np.sqrt(n_obs - 1)) / np.sqrt(denom_sq)
            probabilistic_sharpe = round(float(norm.cdf(z)), 4)

    # Monte Carlo permutation p-value
    # Fraction of randomly shuffled return streams that produce Sharpe >= actual.
    # p < 0.05: edge unlikely due to chance. p > 0.10: edge not proven.
    monte_carlo_p = _monte_carlo_p_value(net_returns, sharpe, bars_per_year)

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
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown_pct": max_drawdown_pct,
            "win_rate_pct": win_rate_pct,
            "profit_factor": profit_factor,
            "num_trades": num_trades,
            "probabilistic_sharpe": probabilistic_sharpe,
            "monte_carlo_p_value": monte_carlo_p,
        }
    }


def compute_kelly_fraction(net_returns: pd.Series) -> float:
    """
    Half-Kelly criterion: optimal per-bar leverage, capped at 5×.
    f* = 0.5 × (mean_return / variance_of_returns)
    A value of 1.0 means the strategy warrants running at full capital.
    Values > 1 suggest leverage; values < 0 suggest the strategy has no edge.
    """
    var_r = net_returns.var()
    if var_r <= 0 or len(net_returns) < 10:
        return 0.0
    full_kelly = float(net_returns.mean()) / float(var_r)
    half_kelly = 0.5 * full_kelly
    return round(max(-1.0, min(5.0, half_kelly)), 4)


def sample_equity_weekly(equity: pd.Series) -> list[dict]:
    weekly = equity.resample("W").last().dropna()
    return [
        {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
        for d, v in weekly.items()
    ]


# ── Private helpers ────────────────────────────────────────────────────────────

def _monte_carlo_p_value(
    net_returns: pd.Series,
    actual_sharpe: float,
    bars_per_year: int,
    n_sims: int = 2000,
) -> float:
    """
    Permutation test: what fraction of randomly shuffled return streams beats
    the actual Sharpe?  Lower is better; p < 0.05 suggests real edge.
    Uses a fixed seed for reproducibility in the JSON.
    """
    if len(net_returns) < 10 or actual_sharpe <= 0:
        return 1.0
    arr = net_returns.values.copy()
    rng = np.random.default_rng(42)
    beat = 0
    for _ in range(n_sims):
        rng.shuffle(arr)
        s = arr.std()
        if s > 0:
            sr = (arr.mean() / s) * np.sqrt(bars_per_year)
            if sr >= actual_sharpe:
                beat += 1
    return round(beat / n_sims, 4)


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
