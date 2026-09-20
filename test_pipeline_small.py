import sys
sys.path.insert(0, 'src')
import pandas as pd 
import glob
import os 
from event_study import build_event_study_dataset

sentiment_path = 'data/processed/news_with_sentiment.csv'
if not os.path.exists(sentiment_path):
    print(f"[ERROR] {sentiment_path} not found. Run 'python run_pipeline.py' at least once. Or just the sentiment.py step to generate it first.")
    sys.exit(1)

sentiment_df = pd.read_csv(sentiment_path, parse_dates=['datetime'])
sentiment_df_small = sentiment_df.head(20) # just first 20 rows for a quick test 

prices = {}
for path in glob.glob('data/raw/prices_*.csv'):
    ticker = os.path.basename(path).replace('prices_', '').replace('.csv', '')
    prices[ticker] = pd.read_csv(path, index_col='Date', parse_dates=True)

print(f"Loaded price data for: {list(prices.keys())}")
print(f"Loaded sentiment data with {len(sentiment_df_small)} rows for testing.")

result = build_event_study_dataset(sentiment_df_small, prices)

print()
if not result.empty:
    print(result[['ticker', 'label', 'confidence', 'CAR', 'market_adjusted']])
else:
    print("[WARN] No usable events produced in this test run. Check date alignment / data coverage.")

print()
print("Done.")