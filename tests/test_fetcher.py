import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from tests.conftest import make_df


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


def test_fetch_ohlcv_falls_back_to_esf_when_nqf_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("data.fetcher.CACHE_DIR", tmp_path)

    def side_effect(ticker, **kwargs):
        return pd.DataFrame() if ticker == "NQ=F" else make_df(250)

    with patch("yfinance.download", side_effect=side_effect):
        from data.fetcher import fetch_ohlcv
        df = fetch_ohlcv("NQ=F", "2018-01-01", "2025-12-31")
    assert len(df) == 250
