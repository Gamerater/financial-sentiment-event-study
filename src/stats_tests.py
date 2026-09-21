"""
stats_tests.py
--------------
Statistical analysis of the event-study dataset. This produces the actual
"results" of your paper: is there a statistically significant relationship
between news sentiment and subsequent abnormal returns?

Three complementary analyses, all standard in event-study papers:

1. GROUP COMPARISON (t-test):
   Split events into positive-sentiment vs negative-sentiment groups.
   Test whether mean CAR differs significantly between the two groups.
   H0: mean CAR(positive events) == mean CAR(negative events)

2. CORRELATION:
   Treat sentiment_score as continuous (-1 to +1) and CAR as continuous.
   Compute Pearson correlation coefficient + p-value.
   This tests degree of association, not just group difference.

3. ONE-SAMPLE TESTS PER GROUP:
   For each sentiment group individually, test whether mean CAR is
   significantly different from zero (i.e., "do positive-sentiment events
   show abnormal returns at all, regardless of the other group?")

Usage:
    python src/stats_tests.py
"""

import os
import datetime as dt
import pandas as pd
import numpy as np
from scipy import stats

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RUN_HISTORY_PATH = os.path.join(RESULTS_DIR, "run_history.csv")


def run_group_ttest(df: pd.DataFrame) -> dict:
    """
    Independent-samples t-test comparing mean CAR of positive-sentiment
    events vs negative-sentiment events. Neutral events are excluded from
    this specific test since we're testing the polarity effect directly.
    """
    pos = df[df["label"] == "positive"]["CAR"].dropna()
    neg = df[df["label"] == "negative"]["CAR"].dropna()

    if len(pos) < 2 or len(neg) < 2:
        return {"error": "Not enough events in one or both groups for a t-test (need >=2 each)."}

    t_stat, p_value = stats.ttest_ind(pos, neg, equal_var=False)  # Welch's t-test, doesn't assume equal variance

    return {
        "n_positive": len(pos),
        "n_negative": len(neg),
        "mean_CAR_positive": pos.mean(),
        "mean_CAR_negative": neg.mean(),
        "std_CAR_positive": pos.std(),
        "std_CAR_negative": neg.std(),
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


def run_correlation(df: pd.DataFrame) -> dict:
    """
    Pearson correlation between continuous sentiment_score (-1 to +1) and
    CAR across ALL events (positive, negative, and neutral together).
    """
    clean = df[["sentiment_score", "CAR"]].dropna()
    if len(clean) < 3:
        return {"error": "Not enough events for correlation (need >=3)."}

    r, p_value = stats.pearsonr(clean["sentiment_score"], clean["CAR"])

    return {
        "n_events": len(clean),
        "pearson_r": r,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


def run_one_sample_tests(df: pd.DataFrame) -> dict:
    """
    For each sentiment label, tests whether mean CAR is significantly
    different from zero (one-sample t-test against mu=0).
    """
    results = {}
    for label in ["positive", "negative", "neutral"]:
        group = df[df["label"] == label]["CAR"].dropna()
        if len(group) < 2:
            results[label] = {"error": "Not enough events (need >=2)."}
            continue
        t_stat, p_value = stats.ttest_1samp(group, popmean=0)
        results[label] = {
            "n": len(group),
            "mean_CAR": group.mean(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant_at_0.05": p_value < 0.05,
        }
    return results


def log_run_history(df, ttest_result, corr_result):
    """
    Appends one row to results/run_history.csv every time the full analysis
    runs, so the dashboard can plot how sample size and statistical results
    have evolved over the course of the data-collection period.

    This is intentionally append-only (never overwritten), so it becomes a
    genuine research log: every pipeline run adds one data point, giving you
    a real trend line by the time you write up results -- e.g. "the
    correlation coefficient stabilized once n exceeded 200 events" is a
    claim you can only make if this history was captured along the way.

    Columns logged:
      run_timestamp   - when this run happened (ISO format)
      n_events        - total events analyzed this run
      n_positive      - positive-sentiment event count
      n_negative      - negative-sentiment event count
      n_neutral       - neutral-sentiment event count
      pearson_r       - correlation coefficient (sentiment vs CAR), all events
      corr_p_value    - p-value for that correlation
      ttest_p_value   - p-value for positive-vs-negative group t-test (None if
                        the t-test couldn't run due to insufficient group size)
    """
    label_counts = df["label"].value_counts()

    row = {
        "run_timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "n_events": len(df),
        "n_positive": int(label_counts.get("positive", 0)),
        "n_negative": int(label_counts.get("negative", 0)),
        "n_neutral": int(label_counts.get("neutral", 0)),
        "pearson_r": corr_result.get("pearson_r"),
        "corr_p_value": corr_result.get("p_value"),
        "ttest_p_value": ttest_result.get("p_value"),  # None if t-test had an "error" key instead
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    row_df = pd.DataFrame([row])

    if os.path.exists(RUN_HISTORY_PATH):
        row_df.to_csv(RUN_HISTORY_PATH, mode="a", header=False, index=False)
    else:
        row_df.to_csv(RUN_HISTORY_PATH, mode="w", header=True, index=False)

    print(f"[history] Logged this run to {RUN_HISTORY_PATH}")


def run_full_analysis(dataset_path: str = None) -> dict:
    """
    Runs all three analyses on the event study dataset and prints a
    human-readable summary. Also returns everything as a dict so it can be
    used programmatically (e.g. in the demo notebook).
    """
    if dataset_path is None:
        dataset_path = os.path.join(PROCESSED_DIR, "event_study_dataset.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No dataset found at {dataset_path}. Run event_study.py first.")

    df = pd.read_csv(dataset_path, parse_dates=["datetime"])
    print(f"Loaded {len(df)} events from {dataset_path}\n")
    print("Label distribution:")
    print(df["label"].value_counts())
    print()

    ttest_result = run_group_ttest(df)
    corr_result = run_correlation(df)
    one_sample_results = run_one_sample_tests(df)

    print("=" * 60)
    print("1. GROUP COMPARISON: Positive vs Negative sentiment events")
    print("=" * 60)
    for k, v in ttest_result.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 60)
    print("2. CORRELATION: sentiment_score vs CAR (all events)")
    print("=" * 60)
    for k, v in corr_result.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 60)
    print("3. ONE-SAMPLE TESTS: is mean CAR different from 0, per group?")
    print("=" * 60)
    for label, res in one_sample_results.items():
        print(f"  [{label}]")
        for k, v in res.items():
            print(f"    {k}: {v}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "statistical_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Event Study Statistical Summary\nEvents analyzed: {len(df)}\n\n")
        f.write("GROUP T-TEST (positive vs negative):\n")
        for k, v in ttest_result.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nCORRELATION (sentiment_score vs CAR):\n")
        for k, v in corr_result.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nONE-SAMPLE TESTS (mean CAR vs 0):\n")
        for label, res in one_sample_results.items():
            f.write(f"  [{label}]\n")
            for k, v in res.items():
                f.write(f"    {k}: {v}\n")
    print(f"\n[saved] Summary written to {summary_path}")

    log_run_history(df, ttest_result, corr_result)

    return {
        "group_ttest": ttest_result,
        "correlation": corr_result,
        "one_sample_tests": one_sample_results,
    }


if __name__ == "__main__":
    run_full_analysis()