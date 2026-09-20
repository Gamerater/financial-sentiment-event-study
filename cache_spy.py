import sys
sys.path.insert(0, 'src')
from fetch_prices import fetch_and_cache_all

# force_refresh=False so your existing 7 cached tickers are NOT re-downloaded
# (saves time/API calls) -- only SPY, which has no cache file yet, gets fetched.
data = fetch_and_cache_all(force_refresh=False)

print("\nCached tickers now available:")
for ticker, df in data.items():
    print(f"  {ticker}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")