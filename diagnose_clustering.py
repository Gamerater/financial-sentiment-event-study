import sys
sys.path.insert(0, 'src')
import pandas as pd
import glob
import os

DATASET_PATH = 'data/processed/event_study_dataset.csv'
df = pd.read_csv(DATASET_PATH, parse_dates=['datetime'])

print("All AAPL events, sorted by date:")
aapl = df[df['ticker'] == 'AAPL'].sort_values('datetime')
print(aapl[['datetime', 'CAR', 'n_headlines']])
print()

# Load AAPL price data to see what trading days actually exist near these event dates
prices = pd.read_csv('data/raw/prices_AAPL.csv', index_col='Date', parse_dates=True)
print(f"AAPL price data available: {prices.index.min().date()} to {prices.index.max().date()}")
print(f"Last 10 trading days in price data:")
print(prices.index[-10:].to_list())
print()

print("For each AAPL event date, showing which trading day it snaps to and the T+1..T+3 window used:")
for _, row in aapl.iterrows():
    event_date = pd.Timestamp(row['datetime']).normalize()
    if event_date not in prices.index:
        future = prices.index[prices.index > event_date]
        snapped = future[0] if len(future) > 0 else None
    else:
        snapped = event_date

    if snapped is not None and snapped in prices.index:
        loc = prices.index.get_loc(snapped)
        window_dates = prices.index[loc+1:loc+4].to_list() if loc+3 < len(prices) else "INSUFFICIENT FUTURE DATA"
    else:
        window_dates = "EVENT DATE NOT FOUND / NO FUTURE DATA"

    print(f"  Event date: {event_date.date()}  ->  snapped to trading day: {snapped}  ->  window: {window_dates}")