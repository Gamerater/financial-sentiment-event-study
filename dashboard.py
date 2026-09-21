"""
dashboard.py
-------------
A local, read-only results dashboard for the event-study project. Reads
whatever the pipeline has already produced (results/statistical_summary.txt,
data/processed/event_study_dataset.csv, results/figures/*.png) and displays
them in a browser -- it does NOT run the pipeline itself. This keeps the
demo fast and network-independent: run `python run_pipeline.py` beforehand
(or during a break in your presentation), then use the dashboard's "Reload
results" button to pick up the latest numbers without re-fetching anything.

Usage:
    python dashboard.py

Then open http://127.0.0.1:5000 in your browser (it should open automatically).
"""

import os
import sys
import webbrowser
import threading
import datetime as dt

from flask import Flask, jsonify, render_template, send_from_directory
import pandas as pd

def find_project_root(start_dir):
    """
    Walks upward from `start_dir` looking for run_pipeline.py, which marks
    the actual project root. This makes the dashboard work correctly whether
    dashboard.py sits at the project root or one level deeper (e.g. inside
    a "dashboard" subfolder), without requiring an exact extraction layout.
    """
    current = start_dir
    for _ in range(5):  # don't walk up forever
        if os.path.exists(os.path.join(current, "run_pipeline.py")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return start_dir  # fallback: assume dashboard.py's own folder is correct


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = find_project_root(SCRIPT_DIR)
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Look for the templates/static folders next to THIS file (dashboard.py),
# regardless of what the current working directory happens to be, or
# whether dashboard.py was placed at the project root or one level deep.
# This makes the app resilient to exactly how the file was extracted/copied,
# rather than requiring an exact folder layout.
TEMPLATE_FOLDER = os.path.join(SCRIPT_DIR, "dashboard", "templates")
STATIC_FOLDER = os.path.join(SCRIPT_DIR, "dashboard", "static")

if not os.path.isdir(TEMPLATE_FOLDER):
    # Fallback: dashboard.py is itself already inside the "dashboard" folder
    # (templates/static are siblings, not inside a nested "dashboard/" subfolder)
    alt_template = os.path.join(SCRIPT_DIR, "templates")
    if os.path.isdir(alt_template):
        TEMPLATE_FOLDER = alt_template
        STATIC_FOLDER = os.path.join(SCRIPT_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)


def parse_statistical_summary(path):
    """
    Parses the plain-text statistical_summary.txt (written by stats_tests.py)
    into a structured dict the frontend can render as clean stat cards,
    rather than dumping raw text at the user.
    """
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    summary = {"events_analyzed": None, "group_ttest": {}, "correlation": {}, "one_sample": {}}
    section = None
    current_label = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Events analyzed:"):
            summary["events_analyzed"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("GROUP T-TEST"):
            section = "group_ttest"
        elif stripped.startswith("CORRELATION"):
            section = "correlation"
        elif stripped.startswith("ONE-SAMPLE TESTS"):
            section = "one_sample"
        elif stripped.startswith("[") and stripped.endswith("]") and section == "one_sample":
            current_label = stripped.strip("[]")
            summary["one_sample"][current_label] = {}
        elif ":" in stripped and section:
            key, _, value = stripped.partition(":")
            key, value = key.strip(), value.strip()
            if section == "one_sample" and current_label is not None:
                summary["one_sample"][current_label][key] = value
            elif section in ("group_ttest", "correlation"):
                summary[section][key] = value

    return summary


def load_event_dataset(path):
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime", ascending=False)
    for col in ["sentiment_score", "confidence", "CAR"]:
        if col in df.columns:
            df[col] = df[col].round(4)
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")


def compute_per_ticker_stats(dataset_path):
    """
    Computes lightweight summary stats PER TICKER directly from the raw
    event_study_dataset.csv, independent of the aggregate statistical_summary.txt
    (which only covers the all-tickers-combined tests). This lets the dashboard
    show a per-stock view without needing stats_tests.py to be re-run with a
    ticker filter -- we just group the already-computed CAR values here.

    Returns a dict {ticker: {n_events, mean_CAR, std_CAR, correlation, p_value}}.
    Correlation/p_value are omitted (None) when a ticker has fewer than 4 events,
    since a Pearson correlation on 3 or fewer points is not meaningful.
    """
    if not os.path.exists(dataset_path):
        return {}

    df = pd.read_csv(dataset_path, parse_dates=["datetime"])
    if df.empty:
        return {}

    from scipy import stats as scipy_stats

    result = {}
    for ticker, group in df.groupby("ticker"):
        entry = {
            "n_events": int(len(group)),
            "mean_CAR": round(float(group["CAR"].mean()), 4),
            "std_CAR": round(float(group["CAR"].std()), 4) if len(group) > 1 else None,
            "date_range": [
                group["datetime"].min().strftime("%Y-%m-%d"),
                group["datetime"].max().strftime("%Y-%m-%d"),
            ],
            "correlation": None,
            "p_value": None,
        }
        if len(group) >= 4:
            r, p = scipy_stats.pearsonr(group["sentiment_score"], group["CAR"])
            entry["correlation"] = round(float(r), 3)
            entry["p_value"] = round(float(p), 4)
        result[ticker] = entry

    return result


def compute_freshness(dataset_path, news_archive_path, fetch_log_path):
    """
    Computes the "data freshness" facts the dashboard shows to reinforce that
    this is a live, ongoing study rather than a one-off snapshot:
      - last_pipeline_run: when event_study_dataset.csv was last written
        (i.e. when run_pipeline.py / stats_tests.py last completed)
      - last_fetch: when the news archive was last updated (may be more
        recent than the last full pipeline run, since daily_fetch.py can run
        independently of run_pipeline.py)
      - days_collecting: whole days between the EARLIEST logged fetch and now,
        read from fetch_log.txt if it exists (falls back to the news
        archive's own earliest headline date if the log is missing)
      - total_fetch_runs: how many times daily_fetch.py / fetch_news.py has
        successfully run, counted from fetch_log.txt

    Every field is None (not an error) when its source file doesn't exist
    yet, so a brand-new project just shows sparser info instead of failing.
    """
    result = {
        "last_pipeline_run": None,
        "last_fetch": None,
        "days_collecting": None,
        "total_fetch_runs": None,
    }

    if os.path.exists(dataset_path):
        mtime = os.path.getmtime(dataset_path)
        result["last_pipeline_run"] = dt.datetime.fromtimestamp(mtime).isoformat(timespec="seconds")

    if os.path.exists(news_archive_path):
        mtime = os.path.getmtime(news_archive_path)
        result["last_fetch"] = dt.datetime.fromtimestamp(mtime).isoformat(timespec="seconds")

    if os.path.exists(fetch_log_path):
        try:
            with open(fetch_log_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                # Each line starts "YYYY-MM-DD HH:MM:SS | ..." -- parse the
                # timestamp off the first line (earliest run) to compute how
                # many days of collection have elapsed.
                first_ts_str = lines[0].split("|")[0].strip()
                first_ts = dt.datetime.strptime(first_ts_str, "%Y-%m-%d %H:%M:%S")
                result["days_collecting"] = max(0, (dt.datetime.now() - first_ts).days)
                result["total_fetch_runs"] = sum(1 for line in lines if "success=True" in line)
        except Exception:
            pass  # malformed log line shouldn't break the whole endpoint

    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/results")
def api_results():
    """
    Returns everything the dashboard needs in one JSON payload: the parsed
    statistics summary, the event table, per-ticker breakdowns, freshness
    info, and which figure files exist.
    """
    summary_path = os.path.join(RESULTS_DIR, "statistical_summary.txt")
    dataset_path = os.path.join(DATA_PROCESSED, "event_study_dataset.csv")
    news_archive_path = os.path.join(BASE_DIR, "data", "raw", "news_archive.csv")
    fetch_log_path = os.path.join(BASE_DIR, "data", "raw", "fetch_log.txt")

    summary = parse_statistical_summary(summary_path)
    events = load_event_dataset(dataset_path)
    per_ticker = compute_per_ticker_stats(dataset_path)
    freshness = compute_freshness(dataset_path, news_archive_path, fetch_log_path)

    figure_files = ["car_boxplot.png", "sentiment_vs_car_scatter.png", "car_timeline.png", "label_distribution.png"]
    available_figures = [f for f in figure_files if os.path.exists(os.path.join(FIGURES_DIR, f))]

    return jsonify({
        "summary": summary,
        "events": events,
        "n_events": len(events),
        "figures": available_figures,
        "has_data": summary is not None,
        "per_ticker": per_ticker,
        "freshness": freshness,
    })


@app.route("/figures/<path:filename>")
def serve_figure(filename):
    return send_from_directory(FIGURES_DIR, filename)


@app.route("/api/history")
def api_history():
    """
    Returns the run-by-run history logged by stats_tests.py's
    log_run_history(), so the dashboard can plot how sample size and
    statistical results have trended over the data-collection period.
    Returns an empty list (not an error) if no history exists yet -- this
    is expected on a fresh project before the pipeline has run more than once.
    """
    history_path = os.path.join(RESULTS_DIR, "run_history.csv")
    if not os.path.exists(history_path):
        return jsonify({"runs": []})

    df = pd.read_csv(history_path)
    records = df.to_dict(orient="records")
    # Replace NaN (e.g. ttest_p_value when the t-test couldn't run that day)
    # with None so it serializes as valid JSON null rather than the invalid
    # bare "NaN" token that jsonify would otherwise produce from a numpy NaN.
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and pd.isna(value):
                record[key] = None
    return jsonify({"runs": records})


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    print("Starting dashboard at http://127.0.0.1:5000 ...")
    print("Press CTRL+C to stop.")
    app.run(debug=False, port=5000)