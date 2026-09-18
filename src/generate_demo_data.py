"""
generate_demo_data.py
----------------------
Generates a realistic SYNTHETIC event-study dataset for demo/testing purposes.

WHY THIS EXISTS:
Free news sources only give a rolling window of recent headlines (see the
warning in fetch_news.py). On any given day you might only capture 10-30
real headlines total across 7 tickers -- not enough to run meaningful
statistics or show an impressive demo on day one.

This script builds a clearly-labeled SYNTHETIC dataset with a realistic,
modest effect size (small correlation, some noise, not a suspiciously
perfect result) so that:
  (a) you can demo the FULL pipeline (stats + charts) working end-to-end
      today, without waiting weeks to accumulate real news, and
  (b) you can develop/debug stats_tests.py and visualize.py against a
      known dataset.

*** IMPORTANT: clearly disclose to your teacher which run used real vs
synthetic data. Label demo outputs accordingly. Run the REAL pipeline
(run_pipeline.py) over 2-4 weeks leading up to submission to accumulate a
real headline archive for your actual paper's results. ***

Usage:
    python src/generate_demo_data.py
"""

import os
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

SAMPLE_HEADLINES = {
    "positive": [
        "{ticker} beats quarterly earnings estimates by wide margin",
        "{ticker} announces stronger than expected revenue growth",
        "Analysts upgrade {ticker} stock on positive outlook",
        "{ticker} unveils breakthrough product to strong reception",
        "{ticker} reports record profit driven by demand surge",
        "{ticker} stock rallies on optimistic guidance",
    ],
    "negative": [
        "{ticker} misses earnings expectations, shares under pressure",
        "{ticker} faces regulatory scrutiny over business practices",
        "Analysts downgrade {ticker} citing weak demand",
        "{ticker} announces layoffs amid cost-cutting measures",
        "{ticker} warns of slower growth in upcoming quarter",
        "{ticker} shares slide after disappointing guidance",
    ],
    "neutral": [
        "{ticker} to present at upcoming investor conference",
        "{ticker} announces routine board meeting schedule",
        "{ticker} files standard quarterly report with SEC",
        "{ticker} holds annual shareholder meeting",
        "{ticker} updates corporate governance policy",
    ],
}


def generate_synthetic_dataset(n_events: int = 300, true_effect_size: float = 0.015,
                                 noise_std: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic event-study dataset with a MODEST, realistic
    correlation between sentiment and CAR -- not a suspiciously perfect
    relationship, to keep the demo methodologically honest.

    true_effect_size: how much mean CAR shifts per unit of sentiment_score
                       (e.g. 0.015 means a fully positive event (+1) shifts
                       mean CAR by +1.5 percentage points on average)
    noise_std: standard deviation of random noise added to CAR (this
               dominates the signal, which is realistic -- real markets
               are noisy and news sentiment is only one of many factors)
    """
    rng = np.random.default_rng(seed)
    records = []

    labels_pool = ["positive", "negative", "neutral"]
    label_weights = [0.35, 0.30, 0.35]  # roughly balanced, slightly more pos/neutral

    base_date = pd.Timestamp("2024-06-01")

    for i in range(n_events):
        ticker = rng.choice(TICKERS)
        label = rng.choice(labels_pool, p=label_weights)
        headline_template = rng.choice(SAMPLE_HEADLINES[label])
        title = headline_template.format(ticker=ticker)

        # Sentiment score: matches label direction with some confidence variation
        if label == "positive":
            sentiment_score = rng.uniform(0.4, 0.99)
        elif label == "negative":
            sentiment_score = rng.uniform(-0.99, -0.4)
        else:
            sentiment_score = rng.uniform(-0.15, 0.15)

        confidence = abs(sentiment_score) if label != "neutral" else rng.uniform(0.5, 0.9)

        # CAR = true_effect_size * sentiment_score + noise
        # This is the DESIGNED relationship we're testing for -- a modest,
        # realistic effect buried in noise, not a toy-perfect correlation.
        car = true_effect_size * sentiment_score + rng.normal(0, noise_std)

        event_date = base_date + pd.Timedelta(days=int(rng.integers(0, 90)))

        records.append({
            "ticker": ticker,
            "datetime": event_date,
            "title": title,
            "label": label,
            "sentiment_score": sentiment_score,
            "confidence": confidence,
            "CAR": car,
        })

    df = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)
    return df


def save_demo_dataset():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df = generate_synthetic_dataset()
    out_path = os.path.join(PROCESSED_DIR, "event_study_dataset_DEMO_SYNTHETIC.csv")
    df.to_csv(out_path, index=False)
    print(f"[demo] Generated {len(df)} synthetic events -> {out_path}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print("\nRemember: this is SYNTHETIC data for pipeline demonstration only.")
    print("Run the real pipeline (run_pipeline.py) to collect actual data for your paper.")
    return df


if __name__ == "__main__":
    save_demo_dataset()
