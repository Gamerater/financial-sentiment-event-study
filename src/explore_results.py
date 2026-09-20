"""
explore_results.py
--------------------
Diagnostic / exploratory script -- NOT part of the core pipeline. Run this
after run_pipeline.py to understand WHY the statistics look the way they do,
before deciding whether to wait for more data or adjust methodology.

This does not change any files in data/ or results/ -- it's purely for
your own understanding.

Usage:
    python src/explore_results.py
"""

import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DATASET_PATH = os.path.join(PROCESSED_DIR, "event_study_dataset.csv")


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: {DATASET_PATH} not found. Run run_pipeline.py first.")
        return

    df = pd.read_csv(DATASET_PATH, parse_dates=["datetime"])
    print(f"Loaded {len(df)} events\n")

    # -----------------------------------------------------------------
    # 1. Biggest movers -- what does the extreme end of CAR look like?
    # -----------------------------------------------------------------
    print("=" * 70)
    print("TOP 10 HIGHEST CAR EVENTS (biggest positive abnormal return)")
    print("=" * 70)
    top = df.nlargest(10, "CAR")[["ticker", "datetime", "label", "confidence", "CAR", "title"]]
    for _, row in top.iterrows():
        print(f"  [{row['label']:8s} conf={row['confidence']:.2f}] CAR={row['CAR']:+.4f}  {row['ticker']:5s}  {row['title'][:70]}")

    print()
    print("=" * 70)
    print("TOP 10 LOWEST CAR EVENTS (biggest negative abnormal return)")
    print("=" * 70)
    bottom = df.nsmallest(10, "CAR")[["ticker", "datetime", "label", "confidence", "CAR", "title"]]
    for _, row in bottom.iterrows():
        print(f"  [{row['label']:8s} conf={row['confidence']:.2f}] CAR={row['CAR']:+.4f}  {row['ticker']:5s}  {row['title'][:70]}")

    # -----------------------------------------------------------------
    # 2. Breakdown by ticker -- is one stock driving/hiding the signal?
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("BREAKDOWN BY TICKER")
    print("=" * 70)
    by_ticker = df.groupby("ticker").agg(
        n_events=("CAR", "count"),
        mean_CAR=("CAR", "mean"),
        std_CAR=("CAR", "std"),
    ).sort_values("n_events", ascending=False)
    print(by_ticker)

    # Correlation per-ticker (only if enough events)
    print("\nPer-ticker sentiment-CAR correlation (only shown if n>=10):")
    for ticker, group in df.groupby("ticker"):
        if len(group) >= 10:
            r = group["sentiment_score"].corr(group["CAR"])
            print(f"  {ticker}: n={len(group)}, r={r:.3f}")

    # -----------------------------------------------------------------
    # 3. Duplicate CAR values -- how many events share an identical CAR?
    #    (this happens when multiple headlines land on the same day/ticker
    #    and therefore share the same event window -- normal, but good to
    #    quantify since it affects "effective" independent sample size)
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("EVENT CLUSTERING (same ticker+CAR = same underlying price move)")
    print("=" * 70)
    dup_groups = df.groupby(["ticker", "CAR"]).size()
    dup_groups = dup_groups[dup_groups > 1].sort_values(ascending=False)
    if len(dup_groups) > 0:
        print(f"Found {len(dup_groups)} (ticker, CAR) combinations shared by multiple headlines:")
        print(dup_groups.head(10))
        total_events_in_clusters = dup_groups.sum()
        print(f"\n{total_events_in_clusters} of {len(df)} events ({100*total_events_in_clusters/len(df):.0f}%) "
              f"belong to a cluster sharing an identical CAR with other headlines.")
        print("This means your EFFECTIVE independent sample size (unique ticker-day price")
        print("moves) is smaller than your headline count -- worth mentioning in your paper's")
        print("limitations, and worth considering whether to aggregate same-day headlines per")
        print("ticker into a single averaged sentiment score per event in a future iteration.")
    else:
        print("No clustering found -- every event has a unique CAR value.")

    # -----------------------------------------------------------------
    # 4. Source breakdown, if available (was this headline from yfinance
    #    or finviz? -- only works if you re-run with source column kept)
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("CONFIDENCE DISTRIBUTION BY LABEL")
    print("=" * 70)
    print(df.groupby("label")["confidence"].describe()[["mean", "min", "max", "count"]])

    print()
    print("Done. Use these patterns to decide: wait for more data, adjust the")
    print("event window, filter by source/ticker, or aggregate same-day events.")


if __name__ == "__main__":
    main()