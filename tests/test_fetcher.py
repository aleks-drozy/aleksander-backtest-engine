import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from tests.conftest import make_df, make_intraday_df


def test_fetch_ohlcv_returns_dataframe(tmp_path, monkeypatch):
    monkeypatch.setattr("data.fetcher.CACHE_DIR", tmp_path)
    with patch("yfinance.download", return_value=make_df(250)):
        from data.fetcher import fetch_ohlcv
        df = fetch_ohlcv("SPY", "2020-01-01", "2022-12-31")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "Close" in df.columns


def test_fetch_ohlcv_caches_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr("data.fetcher.CACHE_DIR", tmp_path)
    with patch("yfinance.download", return_value=make_df(250)) as mock_dl:
        from data.fetcher import fetch_ohlcv
        fetch_ohlcv("SPY", "2020-01-01", "2022-12-31")
        fetch_ohlcv("SPY", "2020-01-01", "2022-12-31")
    assert mock_dl.call_count == 1


def test_cache_roundtrip_preserves_tz_aware_intraday_index(tmp_path, monkeypatch):
    """A cached intraday frame must reload as the same tz-aware DatetimeIndex.

    Regression: yfinance>=1.4 indexes intraday bars in exchange-local time, so a
    720-day window carries both -04:00 and -05:00. The cache could not restore
    mixed offsets, handing back a str Index whose .tzinfo lookup then blew up.
    """
    monkeypatch.setattr("data.fetcher.CACHE_DIR", tmp_path)
    source = make_intraday_df(250)
    assert len({ts.utcoffset() for ts in source.index}) == 2, "fixture must span a DST change"

    with patch("yfinance.download", return_value=source):
        from data.fetcher import fetch_ohlcv
        fetch_ohlcv("NQ=F", "2025-10-30", "2025-11-09", "1h")           # miss -> writes cache
        cached = fetch_ohlcv("NQ=F", "2025-10-30", "2025-11-09", "1h")  # hit  -> reads cache

    assert isinstance(cached.index, pd.DatetimeIndex)
    assert cached.index.tz is not None
    assert list(cached.index) == list(source.index)


def test_cache_roundtrip_keeps_daily_index_tz_naive(tmp_path, monkeypatch):
    """^VIX daily bars arrive tz-naive; the cache must not invent a timezone.

    merge_vix aligns on normalised dates, so localising these would shift them.
    """
    monkeypatch.setattr("data.fetcher.CACHE_DIR", tmp_path)
    source = make_df(250)
    assert source.index.tz is None, "fixture must be tz-naive"

    with patch("yfinance.download", return_value=source):
        from data.fetcher import fetch_ohlcv
        fetch_ohlcv("^VIX", "2020-01-01", "2022-12-31", "1d")
        cached = fetch_ohlcv("^VIX", "2020-01-01", "2022-12-31", "1d")

    assert isinstance(cached.index, pd.DatetimeIndex)
    assert cached.index.tz is None
    assert list(cached.index) == list(source.index)


def test_fetch_ohlcv_raises_on_insufficient_data_never_substitutes(tmp_path, monkeypatch):
    """Insufficient data must raise — never silently swap in another instrument.

    Regression: the fetcher used to fall back to ES=F when NQ=F came up short,
    while run_all.py kept labelling the results NQ=F, so published numbers could
    describe S&P data as Nasdaq. A failed fetch now fails the run instead.
    """
    monkeypatch.setattr("data.fetcher.CACHE_DIR", tmp_path)

    with patch("yfinance.download", return_value=pd.DataFrame()) as mock_dl:
        from data.fetcher import fetch_ohlcv
        with pytest.raises(ValueError, match="NQ=F"):
            fetch_ohlcv("NQ=F", "2018-01-01", "2025-12-31")

    # Only the requested ticker was attempted, and nothing was cached.
    assert mock_dl.call_count == 1
    assert list(tmp_path.glob("*.parquet")) == []
