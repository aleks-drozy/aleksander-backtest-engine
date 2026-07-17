import numpy as np
import pandas as pd
import pytest


def make_df(n: int = 100, start: str = "2020-01-01", daily_return: float = 0.001) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="B")
    close = 100.0 * (1 + daily_return) ** np.arange(n)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.ones(n) * 1_000_000,
        },
        index=dates,
    )


def make_intraday_df(n: int = 250, start: str = "2025-10-30 09:30") -> pd.DataFrame:
    """Hourly bars indexed in exchange-local time, as yfinance>=1.4 returns them.

    Defaults span the 2 Nov EDT->EST change, so the index carries both -04:00 and
    -05:00 offsets — the case that breaks a naive CSV round-trip.
    """
    idx = pd.date_range(start, periods=n, freq="1h", tz="America/New_York", name="Datetime")
    close = 100.0 * (1 + 0.0001) ** np.arange(n)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.ones(n) * 1_000_000,
        },
        index=idx,
    )


@pytest.fixture
def trending_df():
    return make_df(200, daily_return=0.001)


@pytest.fixture
def flat_df():
    return make_df(200, daily_return=0.0)
