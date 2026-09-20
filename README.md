# Correlating Financial News Sentiment with Short-Term Stock Price Movement: An Event-Study Approach

An event-study analysis testing whether financial news sentiment correlates with
short-term abnormal stock returns in the days following a news event — **not** a
stock price prediction model. This distinction matters: predicting prices is an
extremely hard (arguably unsolved) problem, while measuring *correlation* between
sentiment and subsequent movement is an established, publishable methodology used
throughout empirical finance research (see Methodology below).

## Research Question

> Do positive/negative sentiment shifts in financial news headlines correlate
> with statistically significant abnormal stock returns in the following
> trading days?

## Project Structure

```
financial-sentiment-event-study/
├── run_pipeline.py          # runs the entire pipeline end-to-end
├── requirements.txt
├── src/
│   ├── fetch_prices.py       # downloads OHLCV price history (yfinance)
│   ├── fetch_news.py          # fetches news headlines (yfinance + Finviz)
│   ├── sentiment.py            # FinBERT sentiment scoring
│   ├── event_study.py          # abnormal return / CAR calculation (core methodology)
│   ├── stats_tests.py          # t-tests, correlation, hypothesis testing
│   ├── visualize.py             # generates all charts
│   └── generate_demo_data.py    # synthetic dataset for demo/testing (see below)
├── data/
│   ├── raw/                  # cached price + news downloads
│   └── processed/             # sentiment-scored news, final event-study dataset
├── results/
│   ├── figures/                # generated charts (.png)
│   └── statistical_summary.txt
├── notebooks/
│   └── analysis.ipynb          # walkthrough notebook for demo/presentation
└── paper/                    # research paper (later deliverable)
```

## Methodology

This follows the standard **event study** framework used in empirical finance
research (see MacKinlay, 1997, *"Event Studies in Economics and Finance,"*
Journal of Economic Literature):

1. **Event**: a news headline about ticker X published at time T, scored for
   sentiment (positive / negative / neutral) using **FinBERT**, a BERT model
   fine-tuned specifically on financial text (Araci, 2019).
2. **Event window**: trading days T+1 through T+3 (the days *after* the news,
   avoiding same-day ambiguity about whether news broke before or after market
   close).
3. **Confidence filtering**: events where FinBERT's prediction confidence is
   below 0.6 are excluded before analysis, since a low-confidence label (e.g.
   51% "positive") is closer to a coin flip than a real signal and would only
   add noise to the sentiment groups.
4. **Same-day aggregation** (important): multiple headlines about the same
   ticker on the same calendar day are collapsed into ONE event with an
   averaged sentiment score, rather than counted as separate independent
   observations. This matters a lot in practice -- a single major news day
   (e.g. an earnings report) can generate 10-20 headlines that all share the
   exact same subsequent price move, since the stock only moves once
   regardless of article count. Treating each headline as independent would
   be a form of pseudo-replication: it inflates the apparent sample size and
   understates the true uncertainty in any statistical test. Aggregating to
   one event per (ticker, day) gives the statistically honest sample size.
5. **Market-adjusted Abnormal Return (AR)**: `AR_t = Stock_Return_t - SPY_Return_t`
   (SPY, the S&P 500 ETF, as the market proxy). This isolates stock-specific
   movement from broad market-wide moves -- e.g. if the whole market drops 2%
   on unrelated macro news, that shouldn't be attributed to our ticker's own
   news event. This is the standard "market-adjusted return model" in event
   study literature, a middle ground between a naive same-stock baseline and
   a full CAPM/market-model regression with estimated beta.
6. **Cumulative Abnormal Return (CAR)**: sum of AR over the event window.
7. **Hypothesis testing**: Welch's t-test comparing mean CAR between positive-
   and negative-sentiment event groups, plus Pearson correlation between
   continuous sentiment score and CAR.

We use market-adjusted returns (assuming beta = 1) rather than a full CAPM
model with estimated beta, for simplicity and tractability on a zero-budget
student timeline; this is explicitly noted as a simplification/limitation
(see Limitations below).

## Data Sources (all free, zero budget)

| Data | Source | Notes |
|---|---|---|
| Stock prices | `yfinance` | Free, no API key, ~2 years daily OHLCV |
| News headlines | `yfinance` news + Finviz scrape | Free, no key; **rolling window only** (see below) |
| Sentiment model | [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) | Free, pretrained, runs on CPU |

**Tickers studied**: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA (major tech sector)

### ⚠️ Important limitation: news data collection strategy

Free news sources do not provide deep historical headline archives (that kind of
data is normally paywalled — Bloomberg, Refinitiv, etc.). `fetch_news.py` only
captures whatever headlines are *currently* live on Yahoo Finance / Finviz for
each ticker at the time it's run.

**To build an adequate sample size, run `run_pipeline.py` (or just
`fetch_news.py`) repeatedly over several weeks** — each run appends new,
de-duplicated headlines to a running archive (`data/raw/news_archive.csv`).
This is disclosed explicitly here and should be stated in the paper's Data
section as a standard, acceptable constraint for a student project, not a
methodological flaw.

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py
```

This runs all six pipeline stages in order and produces:
- `results/statistical_summary.txt` — the hypothesis test results
- `results/figures/*.png` — all charts

First run will download the FinBERT model (~440MB) from HuggingFace, which is
then cached locally — subsequent runs load it instantly.

## Demo Mode (guaranteed working demo, no waiting for real data)

Since real news accumulation takes time, a **synthetic dataset generator** is
included so the full downstream pipeline (statistics + visualization) can be
demonstrated immediately, without waiting weeks for a real headline archive:

```bash
python src/generate_demo_data.py
python -c "import sys; sys.path.insert(0,'src'); from stats_tests import run_full_analysis; run_full_analysis(dataset_path='data/processed/event_study_dataset_DEMO_SYNTHETIC.csv')"
python -c "import sys; sys.path.insert(0,'src'); from visualize import generate_all_figures; generate_all_figures(dataset_path='data/processed/event_study_dataset_DEMO_SYNTHETIC.csv')"
```

The synthetic dataset is built with a **modest, realistic effect size hidden in
noise** (not a suspiciously perfect correlation) purely to demonstrate that the
statistical pipeline correctly detects a real signal when one exists. **Any
results from this synthetic data must be clearly labeled as such** and are for
pipeline demonstration only — not to be presented as the paper's actual findings.

## Automating Daily Data Collection (recommended if you have weeks before your deadline)

Since news sample size determines statistical power, the single most valuable
thing you can do is let `src/daily_fetch.py` run automatically once a day for
several weeks, rather than remembering to run it manually.

### Windows: Task Scheduler setup

1. Open **Task Scheduler** (search for it in the Start menu).
2. Click **Create Basic Task** (right panel).
3. Name it e.g. `Financial Sentiment Daily Fetch`, click Next.
4. Trigger: choose **Daily**, pick a time (e.g. 9:00 AM), click Next.
5. Action: choose **Start a program**, click Next.
6. **Program/script**: the full path to your Python executable, e.g.
   `C:\Users\YOURNAME\AppData\Local\Programs\Python\Python311\python.exe`
   (find yours by running `where python` in PowerShell)
7. **Add arguments**: `src\daily_fetch.py`
8. **Start in**: the full path to your project folder, e.g.
   `D:\Projects\College Projects\Major Project`
9. Click Finish.

To verify it's working, right-click the task in Task Scheduler and choose
**Run**, then check `data/raw/fetch_log.txt` for a new log line.

### Checking progress anytime

```bash
python src/check_progress.py
```

This shows total headlines collected, per-ticker breakdown, and whether your
sample size is likely large enough yet for meaningful statistics — without
running the slower full pipeline.

### Periodic checkpoints (recommended weekly)

Every week or so, run the full pipeline to see how results evolve as your
sample grows:

```bash
python run_pipeline.py
```

Keep a note of the p-values and effect sizes over time — this progression is
worth mentioning in your paper's discussion (e.g., "with n=180 events at the
4-week mark, results were not significant; with n=450 at 8 weeks...").

## Results (update this section once you have real data)

_Run `run_pipeline.py` with real accumulated news data, then paste key numbers
and figures here before submission._

## Limitations

- News sample size is bounded by free-tier data availability; results should
  be interpreted with appropriate statistical caution regarding sample size.
- The market-adjusted return model (beta assumed = 1) is a simplification vs.
  a full CAPM model with estimated per-stock beta; it corrects for broad
  market-wide moves but not for each stock's individual sensitivity to the
  market.
- Same-day aggregation reduces headline count to unique price-move events,
  which is the statistically correct approach but does mean the "sample
  size" that matters for statistical power is the number of unique
  (ticker, day) events, not the number of headlines collected -- worth
  tracking both numbers and reporting the aggregated count as your true N.
- Headline-level sentiment (not full article body) is used, which may miss
  nuance in longer-form financial reporting.
- Correlation, not causation: this study establishes statistical association,
  not that sentiment *causes* price movement (other confounds — trading volume,
  concurrent macro news, etc. — are not controlled for).

## References

- MacKinlay, A. C. (1997). Event Studies in Economics and Finance. *Journal of
  Economic Literature*, 35(1), 13-39.
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained
  Language Models. arXiv:1908.10063.
- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014).
  Good debt or bad debt: Detecting semantic orientations in economic texts.
  *Journal of the Association for Information Science and Technology*.