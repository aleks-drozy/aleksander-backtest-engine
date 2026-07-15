# aleksander-backtest-engine

[![run backtest](https://github.com/aleks-drozy/aleksander-backtest-engine/actions/workflows/run_backtest.yml/badge.svg)](https://github.com/aleks-drozy/aleksander-backtest-engine/actions/workflows/run_backtest.yml)
[![refresh data](https://github.com/aleks-drozy/aleksander-backtest-engine/actions/workflows/refresh.yml/badge.svg)](https://github.com/aleks-drozy/aleksander-backtest-engine/actions/workflows/refresh.yml)

A vectorised Python backtesting engine that runs a library of nine trading strategies over intraday
NASDAQ-100 futures data, scores each one on a risk-adjusted metrics suite, and validates every result
against a typed schema — all wired to GitHub Actions CI.

Built as a clean, extensible harness for comparing strategies on the same footing: one engine, one
metrics definition, one output contract, so results are comparable and reproducible rather than
per-strategy bespoke.

> Research and education only. Not financial advice and not a live trading system — no broker, no
> order routing, no real capital.

## What it does

- Fetches OHLCV data for `NQ=F` (NASDAQ-100 E-mini futures) at a 1-hour interval via `yfinance`.
- Restricts intraday bars to the US cash session (09:30–16:00 ET) before signalling.
- Runs every registered strategy through a single vectorised `Backtester`.
- Scores each strategy on a shared risk-adjusted metrics suite.
- Validates the combined output against a Pydantic schema and writes `results/backtest_results.json`.

## Strategies

Nine strategies, each a self-contained module implementing a common `BaseStrategy` interface:

| Strategy | Style |
|----------|-------|
| SMA Crossover | Trend following |
| MACD Crossover | Trend / momentum |
| RSI Mean Reversion | Mean reversion |
| Bollinger Mean Reversion | Mean reversion |
| VWAP Mean Reversion | Intraday mean reversion |
| MACD + Bollinger Combo | Momentum + volatility |
| Opening Range Breakout | Intraday breakout |
| Overnight Gap Fade | Gap mean reversion |
| IFVG + CISD | Smart-money / structure (from the FYP strategy) |

## Metrics

Beyond total return, the engine reports a risk-adjusted suite (`engine/metrics.py`):

- **Sharpe** — penalises all volatility.
- **Sortino** — penalises only downside volatility.
- **Max drawdown** and **Calmar** — annualised return per unit of drawdown.
- **Probabilistic Sharpe Ratio** (Bailey & López de Prado, 2014) — the probability the true Sharpe
  exceeds zero given the observed sample, adjusted for the non-normality of returns. It answers "how
  much confidence does this sample size actually earn?" rather than trusting a point estimate.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py      # fetch data, run all 9 strategies, validate, write results/backtest_results.json
pytest                 # run the test suite
```

## Project structure

```
data/         # yfinance OHLCV fetcher
engine/       # Backtester, BaseStrategy interface, metrics
strategies/   # 9 strategy modules
schemas.py    # Pydantic result schema (validates run_all output)
run_all.py    # CLI entry point: fetch -> run -> validate -> write JSON
results/       # backtest_results.json (generated)
tests/        # pytest suite (engine, metrics, schemas, strategies, fetcher)
```

## Extending it

Add a strategy by subclassing `BaseStrategy` in `strategies/`, implementing the signal method, and
registering it in `run_all.py`'s `REGISTRY`. The engine, metrics, schema validation, and tests apply
to it automatically.

## Tests & CI

`pytest` covers the backtester, metrics, schemas, strategies, and data fetcher. Two GitHub Actions
workflows keep the repo live: `run_backtest.yml` runs the suite and regenerates results, and
`refresh.yml` refreshes the data.

## Stack

Python · pandas · NumPy · SciPy · Pydantic · yfinance · pytest · GitHub Actions
