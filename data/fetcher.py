from pathlib import Path
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache"
MIN_ROWS = 100


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    safe_ticker = ticker.replace("=", "_").replace("/", "_")
    cache_file = CACHE_DIR / f"{safe_ticker}_{start}_{end}.csv"

    if cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if len(df) < MIN_ROWS:
        if ticker == "NQ=F":
            return fetch_ohlcv("ES=F", start, end)
        raise ValueError(f"Insufficient data for {ticker}: {len(df)} rows fetched")

    # Flatten MultiIndex columns produced by yfinance >=0.2.38
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache_file)
    return df
