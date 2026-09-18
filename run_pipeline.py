"""
run_pipeline.py
----------------
Runs the entire project pipeline end to end, in order:

    1. fetch_prices.py   -> data/raw/prices_<TICKER>.csv
    2. fetch_news.py      -> data/raw/news_archive.csv
    3. sentiment.py        -> data/processed/news_with_sentiment.csv
    4. event_study.py      -> data/processed/event_study_dataset.csv
    5. stats_tests.py      -> results/statistical_summary.txt
    6. visualize.py         -> results/figures/*.png

This is the single command your professor can run to see the whole project
work end-to-end:

    python run_pipeline.py

Note: step 2 (news fetching) only captures whatever news is currently live
on Yahoo Finance / Finviz -- for a real study with enough sample size, run
this pipeline repeatedly over several days/weeks to build up the news
archive before running steps 3-6. See README.md "Data Collection Strategy".
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import glob
import pandas as pd

from fetch_prices import fetch_and_cache_all
from fetch_news import fetch_all_news
from sentiment import score_news_archive
from event_study import build_event_study_dataset
from stats_tests import run_full_analysis
from visualize import generate_all_figures

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")


def main():
    print("#" * 70)
    print("STEP 1/6: Fetching stock price history")
    print("#" * 70)
    prices = fetch_and_cache_all()

    print("\n" + "#" * 70)
    print("STEP 2/6: Fetching news headlines")
    print("#" * 70)
    fetch_all_news()

    print("\n" + "#" * 70)
    print("STEP 3/6: Running FinBERT sentiment analysis")
    print("#" * 70)
    score_news_archive()

    print("\n" + "#" * 70)
    print("STEP 4/6: Building event-study dataset (computing CAR per event)")
    print("#" * 70)
    sentiment_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "processed", "news_with_sentiment.csv"),
        parse_dates=["datetime"]
    )
    build_event_study_dataset(sentiment_df, prices)

    print("\n" + "#" * 70)
    print("STEP 5/6: Running statistical tests")
    print("#" * 70)
    run_full_analysis()

    print("\n" + "#" * 70)
    print("STEP 6/6: Generating figures")
    print("#" * 70)
    generate_all_figures()

    print("\nDONE. Check results/ for statistical_summary.txt and results/figures/ for charts.")


if __name__ == "__main__":
    main()
