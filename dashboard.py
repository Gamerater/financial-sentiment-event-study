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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/results")
def api_results():
    """
    Returns everything the dashboard needs in one JSON payload: the parsed
    statistics summary, the event table, and which figure files exist.
    """
    summary_path = os.path.join(RESULTS_DIR, "statistical_summary.txt")
    dataset_path = os.path.join(DATA_PROCESSED, "event_study_dataset.csv")

    summary = parse_statistical_summary(summary_path)
    events = load_event_dataset(dataset_path)

    figure_files = ["car_boxplot.png", "sentiment_vs_car_scatter.png", "car_timeline.png", "label_distribution.png"]
    available_figures = [f for f in figure_files if os.path.exists(os.path.join(FIGURES_DIR, f))]

    return jsonify({
        "summary": summary,
        "events": events,
        "n_events": len(events),
        "figures": available_figures,
        "has_data": summary is not None,
    })


@app.route("/figures/<path:filename>")
def serve_figure(filename):
    return send_from_directory(FIGURES_DIR, filename)


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    print("Starting dashboard at http://127.0.0.1:5000 ...")
    print("Press CTRL+C to stop.")
    app.run(debug=False, port=5000)