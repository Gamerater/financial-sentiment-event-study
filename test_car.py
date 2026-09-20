import sys
sys.path.insert(0, 'src')
import pandas as pd
from event_study import get_car_for_event

aapl = pd.read_csv('data/raw/prices_AAPL.csv', index_col='Date', parse_dates=True)
spy = pd.read_csv('data/raw/prices_SPY.csv', index_col='Date', parse_dates=True)

test_date = aapl.index[-15]
print('Testing event date:', test_date.date())

car_adjusted = get_car_for_event(aapl, test_date, benchmark_df=spy)
car_raw = get_car_for_event(aapl, test_date, benchmark_df=None)

print('Market-adjusted CAR:', car_adjusted)
print('Raw (non-adjusted) CAR:', car_raw)
print("Difference (should be roughly SPY's own 3-day cumulative return):", car_raw - car_adjusted)