"""
check_progress.py
-------------------
Quick, lightweight status check you can run anytime over the coming weeks
to see how your news archive is growing -- WITHOUT running the full
sentiment/stats/visualization pipeline (which is slower).
 
Usage:
    python src/check_progress.py
"""

import os 
import pandas as pd 

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
NEWS_ARCHIVE_PATH = os.path.join(RAW_DIR, "news_archive.csv")
LOG_PATH = os.path.join(RAW_DIR, "fetch_log.txt")


def main():
    print("=" * 60)
    print("DATA COLLECTION PROGRESS CHECK")
    print("=" * 60)

    if not os.path.exists(NEWS_ARCHIVE_PATH):
        print(f"No news archive found yet. Run src/fetch_news.py or src/daily_fetch.py first.")
        return

    df = pd.read_csv(NEWS_ARCHIVE_PATH, parse_dates=["datetime"])
    print(f"\nTotal unique headlines collected: {len(df)}")
    print(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    print(f"\nHeadlines per ticker:")
    print(df["ticker"].value_counts())

    # Rough guidance on whether sample size is likely adequate yet
    print("\n" + "-" * 60)
    min_per_ticker = df["ticker"].value_counts().min()
    if len(df) < 100:
        print("STATUS: Still early. Keep collecting -- aim for 300+ total headlines")
        print("        before running full statistical analysis for real conclusions.")
    elif len(df) < 300:
        print("STATUS: Building up nicely. A few more weeks will meanigfully improve statistical power. You can do a checkpoint run now with sentiment.py + event_study.py + stats_tests.py to see early trends, but treat results as preliminary.")
    else:
        print("STATUS: Good sample size. Worth runnin the full pipeline. (sentiment.py -> event_study.py -> stats_tests.py) for a real checkpoint on progress.")

    if os.path.exists(LOG_PATH):
        print("\n" + "-" * 60)
        print("Recent fetch log entries:")
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-5:]:
            print(" " + line.strip())
    else:
        print("\nNo fetch log yet -- this means daily_fetch.py has not been run via the scheduled task yet, or you've only used fetch_news.py directly")

if __name__ == "__main__":
    main()