"""
fetch_news.py
-------------
Fetches recent news headlines for each ticker.

Primary source: yfinance's Ticker.news (free, no key, but only ~recent weeks of news
                 per ticker at any given time -- it's a rolling window, not deep history)
Backup source:   Finviz ticker news page (free, scraped HTML, gives a bit more headline
                 history and is a good fallback if yfinance news is empty for a ticker)

IMPORTANT (read this):
Free news sources do NOT give you 2 years of headline history for free -- that kind of
historical news archive is normally paywalled (e.g. Refinitiv, Bloomberg terminals).
This is a known, acceptable limitation for a student project: we will run this script
REPEATEDLY over the coming days/weeks to build up our own headline archive over time,
OR treat this as a shorter-window study (e.g. last 30-60 days) which is still a
methodologically valid event study -- just with a smaller N.
State this limitation explicitly in your paper's "Data" section; it's normal, not a flaw.

Usage:
    python src/fetch_news.py
"""

import os
import time
import json
import datetime as dt
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
           "NFLX", "AMD", "INTC", "ORCL", "CRM", "ADBE"]
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
NEWS_ARCHIVE_PATH = os.path.join(RAW_DIR, "news_archive.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def fetch_yfinance_news(ticker: str, timeout: int = 15) -> pd.DataFrame:
    """
    Pulls current news items from yfinance for one ticker.
    Returns a DataFrame with columns: ticker, datetime, title, publisher, link, source

    IMPORTANT: different yfinance versions expose news fetching differently, and
    not all versions' get_news() accept a `timeout` kwarg even when the method
    exists. We try the newer signature first and gracefully fall back to the
    plain call (and then to the `.news` property) rather than erroring out --
    this keeps the script portable across yfinance versions without needing to
    pin an exact version.
    """
    t = yf.Ticker(ticker)
    items = None

    if hasattr(t, "get_news"):
        try:
            items = t.get_news(timeout=timeout)
        except TypeError:
            # This installed version's get_news() doesn't accept `timeout`
            try:
                items = t.get_news()
            except Exception:
                items = None

    if items is None:
        items = t.news

    items = items or []

    rows = []
    for item in items:
        # yfinance news structure has changed across versions; handle both
        # older flat format and newer nested "content" format defensively.
        content = item.get("content", item)  # fall back to item itself if no "content" key

        title = content.get("title") or item.get("title")
        publisher = (content.get("provider") or {}).get("displayName") if isinstance(content.get("provider"), dict) else item.get("publisher")
        link = None
        if isinstance(content.get("clickThroughUrl"), dict):
            link = content["clickThroughUrl"].get("url")
        link = link or item.get("link")

        pub_date = content.get("pubDate") or item.get("providerPublishTime")
        if isinstance(pub_date, (int, float)):
            pub_dt = dt.datetime.fromtimestamp(pub_date)
        elif isinstance(pub_date, str):
            try:
                pub_dt = pd.to_datetime(pub_date).tz_localize(None)
            except Exception:
                pub_dt = None
        else:
            pub_dt = None

        if title and pub_dt is not None:
            rows.append({
                "ticker": ticker,
                "datetime": pub_dt,
                "title": title,
                "publisher": publisher or "unknown",
                "link": link,
                "source": "yfinance"
            })

    return pd.DataFrame(rows)


def fetch_finviz_news(ticker: str) -> pd.DataFrame:
    """
    Scrapes the Finviz ticker news table as a backup/supplementary source.
    Finviz shows headlines with dates for the last several days per ticker, free, no key.
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    news_table = soup.find("table", {"id": "news-table"})
    if news_table is None:
        return pd.DataFrame()

    rows = []
    last_date = None
    for row in news_table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        date_text = cells[0].get_text(strip=True)
        link_tag = cells[1].find("a")
        if link_tag is None:
            continue
        title = link_tag.get_text(strip=True)
        link = link_tag.get("href")

        # Finviz format: date only appears on the first row of a new day,
        # subsequent rows on the same day show only the time (e.g. "08:15AM")
        if len(date_text.split()) == 2:
            last_date = date_text.split()[0]
            time_text = date_text.split()[1]
        else:
            time_text = date_text

        if last_date is None:
            continue

        try:
            pub_dt = pd.to_datetime(f"{last_date} {time_text}")
        except Exception:
            continue

        rows.append({
            "ticker": ticker,
            "datetime": pub_dt,
            "title": title,
            "publisher": "finviz-listed",
            "link": link,
            "source": "finviz"
        })

    return pd.DataFrame(rows)


def fetch_all_news(tickers=TICKERS, use_finviz_backup: bool = True) -> pd.DataFrame:
    """
    Fetches news for all tickers from all sources, combines, and de-duplicates.
    Appends to a running archive CSV so repeated runs over time build up history.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    all_frames = []

    for ticker in tickers:
        print(f"[fetch] yfinance news for {ticker} ...")
        try:
            df_yf = fetch_yfinance_news(ticker)
            print(f"    -> {len(df_yf)} items")
            all_frames.append(df_yf)
        except Exception as e:
            print(f"    [ERROR] yfinance news failed for {ticker}: {e}")

        if use_finviz_backup:
            print(f"[fetch] finviz news for {ticker} ...")
            try:
                df_fv = fetch_finviz_news(ticker)
                print(f"    -> {len(df_fv)} items")
                all_frames.append(df_fv)
            except Exception as e:
                print(f"    [ERROR] finviz news failed for {ticker}: {e}")

        time.sleep(1.5)  # be polite between tickers

    new_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

    if new_df.empty:
        print("[WARN] No news fetched at all this run.")
        return new_df

    # Merge with existing archive so we accumulate history across multiple runs
    if os.path.exists(NEWS_ARCHIVE_PATH):
        old_df = pd.read_csv(NEWS_ARCHIVE_PATH, parse_dates=["datetime"])
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # De-duplicate on (ticker, title) -- same headline seen twice across runs/sources
    combined = combined.drop_duplicates(subset=["ticker", "title"]).sort_values("datetime")
    combined.to_csv(NEWS_ARCHIVE_PATH, index=False)
    print(f"\n[saved] Archive now has {len(combined)} total unique headlines -> {NEWS_ARCHIVE_PATH}")

    return combined


if __name__ == "__main__":
    df = fetch_all_news()
    if not df.empty:
        print("\nPer-ticker headline counts:")
        print(df["ticker"].value_counts())