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


@pytest.fixture
def trending_df():
    return make_df(200, daily_return=0.001)


@pytest.fixture
def flat_df():
    return make_df(200, daily_return=0.0)
