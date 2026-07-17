from pathlib import Path
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache"
MIN_ROWS = 100


def fetch_ohlcv(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    safe_ticker = ticker.replace("=", "_").replace("/", "_")
    # Parquet, not CSV: intraday bars are indexed in exchange-local time, so a window
    # spanning a DST change holds mixed UTC offsets. CSV round-trips those through
    # strings and pandas hands back a str Index; parquet keeps the tz in its schema.
    cache_file = CACHE_DIR / f"{safe_ticker}_{start}_{end}_{interval}.parquet"

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)

    if len(df) < MIN_ROWS:
        if ticker == "NQ=F":
            return fetch_ohlcv("ES=F", start, end, interval)
        raise ValueError(f"Insufficient data for {ticker}: {len(df)} rows fetched")

    # Flatten MultiIndex columns produced by yfinance >=0.2.38
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_parquet(cache_file)
    return df
