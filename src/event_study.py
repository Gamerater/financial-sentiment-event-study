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

3. NORMAL RETURN: what the stock would be expected to return absent the event.
   We estimate this simply and defensibly as the stock's own trailing mean
   daily return over an estimation window BEFORE the event (e.g. the 60
   trading days before T). This is the "constant mean return model," one of
   the standard, well-documented approaches in event study literature
   (MacKinlay, 1997, "Event Studies in Economics and Finance").

4. ABNORMAL RETURN (AR): AR_t = Actual_Return_t - Normal_Return
   This isolates the part of the day's return NOT explained by the stock's
   own typical behavior -- closer to "what did the news actually seem to move."

5. CUMULATIVE ABNORMAL RETURN (CAR): sum of AR over the event window days.
   CAR = AR_(T+1) + AR_(T+2) + AR_(T+3)
   This is the number we correlate against sentiment score per event.

We deliberately do NOT use a market-model / CAPM-beta approach (regressing
against S&P 500 returns) because that requires more data and adds complexity
without being necessary to answer the paper's core question. The constant
mean return model is simpler, well-precedented, and appropriate to disclose
as a limitation/simplification in the paper.
"""

import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

EVENT_WINDOW = [1, 2, 3]      # trading days after the event to sum returns over (T+1..T+3)
ESTIMATION_WINDOW = 60        # trading days before the event used to estimate "normal" return


def get_normal_return(price_df: pd.DataFrame, event_date: pd.Timestamp,
                       estimation_window: int = ESTIMATION_WINDOW) -> float:
    """
    Estimates the 'normal' (expected, absent-the-event) daily return for a
    stock as of event_date, using its own trailing mean return over the
    prior `estimation_window` trading days.

    Returns NaN if there isn't enough prior data (e.g. event too close to
    the start of our price history) -- these events get dropped later.
    """
    prior_data = price_df[price_df.index < event_date]
    if len(prior_data) < estimation_window:
        return np.nan
    return prior_data["Return"].tail(estimation_window).mean()


def get_car_for_event(price_df: pd.DataFrame, event_date: pd.Timestamp,
                       event_window: list = EVENT_WINDOW) -> float:
    """
    Computes the Cumulative Abnormal Return over `event_window` trading days
    following event_date for one stock.

    Steps:
      1. Find the trading days at event_date + 1, +2, +3 (using the price
         DataFrame's own index, so weekends/holidays are naturally skipped --
         we don't assume exactly N calendar days later, we take the Nth
         trading day AFTER the event in the actual price series).
      2. Estimate the normal return from history before event_date.
      3. AR on each of those days = actual return - normal return.
      4. CAR = sum of the ARs.

    Returns NaN if the event date isn't in range or there isn't enough
    subsequent price data (e.g. event too recent to have T+3 data yet).
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

    normal_return = get_normal_return(price_df, event_date)
    if np.isnan(normal_return):
        return np.nan

    ars = []
    for offset in event_window:
        day_return = price_df["Return"].iloc[event_loc + offset]
        ars.append(day_return - normal_return)

    return float(np.sum(ars))


def build_event_study_dataset(sentiment_df: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """
    Main pipeline function: takes the sentiment-scored news DataFrame and a
    dict of {ticker: price_df}, and computes CAR for every single news event.

    Returns a DataFrame with one row per news event, including:
        ticker, datetime, title, label, sentiment_score, CAR
    Rows where CAR couldn't be computed (insufficient data) are dropped.
    """
    records = []

    for _, row in sentiment_df.iterrows():
        ticker = row["ticker"]
        if ticker not in prices:
            continue
        price_df = prices[ticker]

        event_date = pd.Timestamp(row["datetime"]).normalize()
        car = get_car_for_event(price_df, event_date)

        if not np.isnan(car):
            records.append({
                "ticker": ticker,
                "datetime": row["datetime"],
                "title": row["title"],
                "label": row["label"],
                "sentiment_score": row["sentiment_score"],
                "confidence": row["confidence"],
                "CAR": car,
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
