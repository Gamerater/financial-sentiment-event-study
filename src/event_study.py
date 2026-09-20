"""
event_study.py
--------------
Core event-study methodology. This is the heart of the project.

Concepts (explained for the paper's methodology section too):

1. EVENT: a scored news headline for ticker X published at time T.

2. EVENT WINDOW: the days around T we examine, e.g. [-1, 0, +1, +2, +3].
   We use T+1 to T+3 as our primary window (not T=0 itself) because for most
   headlines we don't have intraday timestamps precise enough to know if the
   news broke before or after market close on day T -- using the *following*
   days avoids "look-ahead" ambiguity and better tests the paper's actual
   question: does sentiment correlate with SUBSEQUENT movement.

3. MARKET-ADJUSTED ABNORMAL RETURN (AR): AR_t = Stock_Return_t - Market_Return_t
   where Market_Return_t is SPY's (S&P 500 ETF) return on the same day. This
   isolates stock-specific movement from broad market-wide moves -- e.g. if
   the whole market drops 2% on macro news unrelated to our story, that
   shouldn't be attributed to our ticker's own news event. This is the
   standard "market-adjusted return model" in event study literature
   (MacKinlay, 1997, "Event Studies in Economics and Finance"), a middle
   ground between the simpler constant-mean-return model and a full
   CAPM/market-model regression with estimated beta.

   NOTE ON METHODOLOGY EVOLUTION: an earlier version of this pipeline used
   the constant-mean-return model (AR = stock's own return minus its own
   trailing historical average). We upgraded to market-adjusted returns
   because the constant-mean model doesn't distinguish "this stock moved
   because of OUR news event" from "this stock moved because the whole
   market moved that day" -- a real confound the market-adjusted model
   corrects for. If you're comparing results across this change in your
   paper, note the methodology difference explicitly.

4. CUMULATIVE ABNORMAL RETURN (CAR): sum of AR over the event window days.
   CAR = AR_(T+1) + AR_(T+2) + AR_(T+3)
   This is the number we correlate against sentiment score per event.

We still do NOT use a full CAPM/market-model regression (estimating each
stock's beta against the market) because that requires more data and
statistical machinery without being necessary to answer the paper's core
question. Market-adjusted return (beta assumed = 1) is simpler, still
well-precedented, and this simplification is disclosed as a limitation.
"""

import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

EVENT_WINDOW = [1, 2, 3]      # trading days after the event to sum returns over (T+1..T+3)
BENCHMARK_TICKER = "SPY"      # market proxy for market-adjusted abnormal returns
MIN_SENTIMENT_CONFIDENCE = 0.6  # events with FinBERT confidence below this are excluded (see filter_by_confidence)


def get_car_for_event(price_df: pd.DataFrame, event_date: pd.Timestamp,
                       benchmark_df: pd.DataFrame = None,
                       event_window: list = EVENT_WINDOW) -> float:
    """
    Computes the Cumulative Abnormal Return over `event_window` trading days
    following event_date for one stock, using MARKET-ADJUSTED abnormal
    returns when a benchmark (SPY) price series is provided.

    Steps:
      1. Find the trading days at event_date + 1, +2, +3 (using the price
         DataFrame's own index, so weekends/holidays are naturally skipped --
         we don't assume exactly N calendar days later, we take the Nth
         trading day AFTER the event in the actual price series).
      2. AR on each of those days = stock's return - benchmark's return on
         the same day (market-adjusted). If no benchmark_df is given, falls
         back to raw (non-market-adjusted) returns -- kept for backward
         compatibility / comparison, but not the recommended default.
      3. CAR = sum of the ARs.

    Returns NaN if the event date isn't in range, there isn't enough
    subsequent price data, or the benchmark is missing matching dates.
    """
    if event_date not in price_df.index:
        # If the event happened on a non-trading day/time, snap to the next
        # available trading day in the price series.
        future_dates = price_df.index[price_df.index > event_date]
        if len(future_dates) == 0:
            return np.nan
        event_date = future_dates[0]

    event_loc = price_df.index.get_loc(event_date)
    max_offset = max(event_window)

    if event_loc + max_offset >= len(price_df):
        return np.nan  # not enough future data yet

    ars = []
    for offset in event_window:
        day_date = price_df.index[event_loc + offset]
        stock_return = price_df["Return"].iloc[event_loc + offset]

        if benchmark_df is not None:
            if day_date not in benchmark_df.index:
                return np.nan  # can't market-adjust without a matching benchmark date
            market_return = benchmark_df.loc[day_date, "Return"]
            ar = stock_return - market_return
        else:
            # Fallback: no benchmark provided, use raw return (not market-adjusted)
            ar = stock_return

        ars.append(ar)

    return float(np.sum(ars))


def filter_by_confidence(sentiment_df: pd.DataFrame,
                          min_confidence: float = MIN_SENTIMENT_CONFIDENCE) -> pd.DataFrame:
    """
    Drops news events where FinBERT's prediction confidence is below
    `min_confidence`. Low-confidence predictions (e.g. FinBERT is only 45%
    sure a headline is "positive" -- barely better than a coin flip) add
    noise to the positive/negative buckets without adding real signal.

    This is applied BEFORE building the event study dataset, so filtered-out
    events never get a CAR computed for them (saves computation and keeps
    the final dataset clean).
    """
    before = len(sentiment_df)
    filtered = sentiment_df[sentiment_df["confidence"] >= min_confidence].copy()
    after = len(filtered)
    print(f"[event_study] Confidence filter (>= {min_confidence}): kept {after}/{before} events "
          f"({before - after} dropped as low-confidence)")
    return filtered


def build_event_study_dataset(sentiment_df: pd.DataFrame, prices: dict,
                                apply_confidence_filter: bool = True,
                                use_market_adjustment: bool = True) -> pd.DataFrame:
    """
    Main pipeline function: takes the sentiment-scored news DataFrame and a
    dict of {ticker: price_df} (which should include BENCHMARK_TICKER's
    price data if use_market_adjustment=True), and computes CAR for every
    news event.

    Returns a DataFrame with one row per news event, including:
        ticker, datetime, title, label, sentiment_score, CAR
    Rows where CAR couldn't be computed (insufficient data) are dropped.
    """
    if apply_confidence_filter:
        sentiment_df = filter_by_confidence(sentiment_df)

    benchmark_df = None
    if use_market_adjustment:
        if BENCHMARK_TICKER in prices:
            benchmark_df = prices[BENCHMARK_TICKER]
        else:
            print(f"[event_study] WARNING: {BENCHMARK_TICKER} price data not found in `prices`. "
                  f"Falling back to non-market-adjusted returns. Run fetch_prices.py to include the benchmark.")

    records = []

    for _, row in sentiment_df.iterrows():
        ticker = row["ticker"]
        if ticker not in prices:
            continue
        if ticker == BENCHMARK_TICKER:
            continue  # never treat the benchmark itself as a study event
        price_df = prices[ticker]

        event_date = pd.Timestamp(row["datetime"]).normalize()
        car = get_car_for_event(price_df, event_date, benchmark_df=benchmark_df)

        if not np.isnan(car):
            records.append({
                "ticker": ticker,
                "datetime": row["datetime"],
                "title": row["title"],
                "label": row["label"],
                "sentiment_score": row["sentiment_score"],
                "confidence": row["confidence"],
                "CAR": car,
                "market_adjusted": benchmark_df is not None,
            })

    result_df = pd.DataFrame(records)
    if not result_df.empty:
        out_path = os.path.join(PROCESSED_DIR, "event_study_dataset.csv")
        result_df.to_csv(out_path, index=False)
        print(f"[event_study] Built dataset with {len(result_df)} usable events -> {out_path}")
    else:
        print("[event_study] WARNING: no usable events produced. Check date alignment / data coverage.")

    return result_df


if __name__ == "__main__":
    # This assumes fetch_prices.py and sentiment.py have already been run.
    import glob

    sentiment_path = os.path.join(PROCESSED_DIR, "news_with_sentiment.csv")
    if not os.path.exists(sentiment_path):
        raise FileNotFoundError("Run sentiment.py first to produce news_with_sentiment.csv")

    sentiment_df = pd.read_csv(sentiment_path, parse_dates=["datetime"])

    prices = {}
    for path in glob.glob(os.path.join(RAW_DIR, "prices_*.csv")):
        ticker = os.path.basename(path).replace("prices_", "").replace(".csv", "")
        prices[ticker] = pd.read_csv(path, index_col="Date", parse_dates=True)

    build_event_study_dataset(sentiment_df, prices)