"""
visualize.py
------------
Produces the standard charts expected in an event-study paper/presentation:

1. CAR distribution by sentiment group (boxplot) -- the main "money chart"
   for your demo: shows visually whether positive/negative events separate.

2. Average cumulative abnormal return over the event window, per sentiment
   group (line chart from T+1 to T+3) -- the classic event-study time chart.

3. Sentiment score vs CAR scatter plot with trend line -- visualizes the
   correlation result directly.

4. Sentiment label distribution (bar chart) -- just a sanity/overview chart.

Usage:
    python src/visualize.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")

COLORS = {"positive": "#2ca02c", "negative": "#d62728", "neutral": "#7f7f7f"}


def plot_car_boxplot(df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["negative", "neutral", "positive"]
    data = [df[df["label"] == lab]["CAR"].dropna().values for lab in order]
    colors = [COLORS[lab] for lab in order]

    bp = ax.boxplot(data, labels=order, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Cumulative Abnormal Return (T+1 to T+3)")
    ax.set_xlabel("News Sentiment")
    ax.set_title("Abnormal Stock Returns by News Sentiment Category")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_car_timeline(df: pd.DataFrame, save_path: str, event_window=(1, 2, 3)):
    """
    NOTE: this uses the per-event CAR total split evenly for illustration
    of shape is NOT accurate day-by-day -- for a true day-by-day AR timeline
    you'd need to store per-day AR, not just the summed CAR. This function
    instead shows mean CAR-so-far is approximated by assuming linear
    accumulation, which is a simplification. See NOTE in code comments of
    event_study.py: a future improvement is to store per-day ARs individually.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in ["positive", "negative", "neutral"]:
        group = df[df["label"] == label]
        mean_total_car = group["CAR"].mean()
        # Simplified linear interpolation across the window for visualization only
        n_days = len(event_window)
        approx_path = [mean_total_car * (i + 1) / n_days for i in range(n_days)]
        ax.plot([0] + list(event_window), [0] + approx_path, marker="o",
                 label=label, color=COLORS[label])

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Trading Days After Event (T)")
    ax.set_ylabel("Mean Cumulative Abnormal Return (approx. path)")
    ax.set_title("Mean CAR Path by Sentiment Group (illustrative)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_scatter_correlation(df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    clean = df[["sentiment_score", "CAR", "label"]].dropna()
    colors = clean["label"].map(COLORS)
    ax.scatter(clean["sentiment_score"], clean["CAR"], c=colors, alpha=0.6, edgecolor="k", linewidth=0.3)

    # Trend line
    if len(clean) >= 2:
        z = np.polyfit(clean["sentiment_score"], clean["CAR"], 1)
        x_line = np.linspace(clean["sentiment_score"].min(), clean["sentiment_score"].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), color="black", linestyle="--", linewidth=1.5, label="Trend")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Sentiment Score (-1 = very negative, +1 = very positive)")
    ax.set_ylabel("Cumulative Abnormal Return")
    ax.set_title("Sentiment Score vs Abnormal Return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_label_distribution(df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts().reindex(["negative", "neutral", "positive"])
    colors = [COLORS[lab] for lab in counts.index]
    ax.bar(counts.index, counts.values, color=colors, alpha=0.7)
    ax.set_ylabel("Number of Events")
    ax.set_title("Distribution of News Sentiment Labels")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {save_path}")


def generate_all_figures(dataset_path: str = None):
    if dataset_path is None:
        dataset_path = os.path.join(PROCESSED_DIR, "event_study_dataset.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No dataset found at {dataset_path}. Run event_study.py first.")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(dataset_path, parse_dates=["datetime"])

    plot_car_boxplot(df, os.path.join(FIGURES_DIR, "car_boxplot.png"))
    plot_car_timeline(df, os.path.join(FIGURES_DIR, "car_timeline.png"))
    plot_scatter_correlation(df, os.path.join(FIGURES_DIR, "sentiment_vs_car_scatter.png"))
    plot_label_distribution(df, os.path.join(FIGURES_DIR, "label_distribution.png"))
    print("\nAll figures generated in", FIGURES_DIR)


if __name__ == "__main__":
    generate_all_figures()
