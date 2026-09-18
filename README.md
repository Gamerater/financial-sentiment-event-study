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
3. **Normal return**: estimated using the constant mean return model — each
   stock's own trailing 60-trading-day average daily return, computed *before*
   the event.
4. **Abnormal Return (AR)**: `AR_t = Actual_Return_t - Normal_Return`
5. **Cumulative Abnormal Return (CAR)**: sum of AR over the event window.
6. **Hypothesis testing**: Welch's t-test comparing mean CAR between positive-
   and negative-sentiment event groups, plus Pearson correlation between
   continuous sentiment score and CAR.

We use the constant mean return model rather than a market/CAPM-beta model for
simplicity and tractability on a zero-budget student timeline; this is
explicitly noted as a simplification/limitation (see Limitations below).

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

## Results (update this section once you have real data)

_Run `run_pipeline.py` with real accumulated news data, then paste key numbers
and figures here before submission._

## Limitations

- News sample size is bounded by free-tier data availability; results should
  be interpreted with appropriate statistical caution regarding sample size.
- The constant mean return model is a simplification vs. a full market/CAPM
  model; it does not control for broad market-wide moves on the event day.
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
