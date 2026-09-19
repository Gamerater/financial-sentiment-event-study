"""
daily_fetch.py
---------------
Lightweight script meant to be run automatically once a day (via Windows
Task Scheduler / cron) to accumulate news headlines over time.
 
Unlike run_pipeline.py (which runs the ENTIRE pipeline including the slow
FinBERT sentiment scoring and stats), this script only does the fast,
cheap step: fetching new news headlines and appending them to the archive.
 
Why separate this out:
- News accumulation needs to happen daily/frequently to build sample size.
- Sentiment scoring + stats + figures should happen occasionally (e.g. weekly)
  when you actually want to check progress, not every single day.
- This keeps the daily automated task fast and lightweight.
 
This script also logs each run (with timestamp + how many new headlines were
added) to data/raw/fetch_log.txt so you can track accumulation progress
over the following weeks without re-reading full console output.
 
Usage (manual):
    python src/daily_fetch.py
 
Usage (scheduled): see README section "Automating Daily Data Collection"
"""

import sys 
import os 
import datetime as dt

sys.path.insert(0, os.path.dirname(__file__))

from fetch_news import fetch_all_news, NEWS_ARCHIVE_PATH
from fetch_news import fetch_and_cache_all

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "fetch_log.txt")

def count_archive_rows():
    if not os.path.exists(NEWS_ARCHIVE_PATH):
        return 0
    with open(NEWS_ARCHIVE_PATH, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1 #minus header row 

def main():
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Starting daily fetch...")

    count_before = count_archive_rows()

    try:
        #Refresh prices too 
        fetch_and_cache_all(force_refresh=False)
        fetch_all_news()
        success = True
        error_msg = ""
    except Exception as e:
        success = False
        error_msg = str(e)
        print(f"[ERROR] Daily fetch failed: {e}")

    count_after = count_archive_rows()
    new_rows = count_after - count_before

    log_line = (
        f"{timestamp} | success = {success} | total_headlines = {count_after} | "
        f"new_this_run={new_rows}"
    )
    if not success:
        log_line += f"{log_line} | error={error_msg}"

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

    print(f"\n{log_line}")
    print(f"Log appended to {LOG_PATH}")

if __name__ == "__main__":
    main()