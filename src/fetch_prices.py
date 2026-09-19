"""
fetch_prices.py
----------------
Downloads daily OHLCV price history for a list of tickers using yfinance
and caches it locally as CSV so we don't re-download every time.

Why we need this:
For the event study, we need daily returns around each news event date.
yfinance gives us free historical price data with no API key required.

Usage:
    python src/fetch_prices.py
"""

import os
import time
import pandas as pd
import yfinance as yf

# ---- CONFIG ----
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
PERIOD = "2y"          # how far back to pull daily prices
INTERVAL = "1d"        # daily bars (enough resolution for a short-term event study)
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def fetch_price_history(ticker: str, period: str = PERIOD, interval: str = INTERVAL,
                         timeout: int = 15) -> pd.DataFrame:
    """
    Fetch OHLCV history for one ticker.
    Returns a DataFrame indexed by Date with columns: Open, High, Low, Close, Volume

    IMPORTANT: yfinance does not apply a network timeout by default, which means
    a stalled connection (common when running unattended via Task Scheduler with
    no one around to notice/retry) can hang the whole script forever. We pass an
    explicit `timeout` (seconds) so a bad connection raises an error instead of
    hanging indefinitely.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval=interval, auto_adjust=True, timeout=timeout)
    if hist.empty:
        raise ValueError(f"No price data returned for {ticker}. Check ticker symbol or internet connection.")

    # Keep only what we need, and make sure the index is timezone-naive
    # (yfinance returns tz-aware timestamps; we strip tz for easier joining with news dates later)
    hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    hist.index = hist.index.tz_localize(None)
    hist.index.name = "Date"
    return hist


def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'Return' column = simple daily % return based on Close price.
    Return_t = (Close_t - Close_{t-1}) / Close_{t-1}
    """
    price_df = price_df.copy()
    price_df["Return"] = price_df["Close"].pct_change()
    return price_df


def fetch_and_cache_all(tickers=TICKERS, force_refresh: bool = False) -> dict:
    """
    Fetches price history for every ticker in `tickers`, computes returns,
    and saves each to data/raw/prices_<TICKER>.csv

    Returns a dict {ticker: DataFrame} for immediate use in this session too.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    all_data = {}

    for ticker in tickers:
        out_path = os.path.join(RAW_DIR, f"prices_{ticker}.csv")

        if os.path.exists(out_path) and not force_refresh:
            print(f"[cache] Loading cached prices for {ticker} from {out_path}")
            df = pd.read_csv(out_path, index_col="Date", parse_dates=True)
        else:
            print(f"[fetch] Downloading price history for {ticker} ...")
            try:
                df = fetch_price_history(ticker)
                df = compute_daily_returns(df)
                df.to_csv(out_path)
                print(f"[fetch]   -> saved {len(df)} rows to {out_path}")
            except Exception as e:
                print(f"[ERROR] Failed to fetch {ticker}: {e}")
                continue
            # Be polite to Yahoo's servers between requests
            time.sleep(1)

        all_data[ticker] = df

    return all_data


if __name__ == "__main__":
    data = fetch_and_cache_all()
    print("\nSummary:")
    for ticker, df in data.items():
        print(f"  {ticker}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")