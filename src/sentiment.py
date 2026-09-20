"""
sentiment.py
------------
Runs FinBERT (a BERT model pretrained specifically on financial text) over
news headlines to produce a sentiment label and score for each one.

Why FinBERT instead of a generic sentiment model (e.g. VADER, TextBlob)?
Generic sentiment models are trained on product reviews / social media and
misread financial language constantly. E.g. "shares plunge" reads as neutral
to a generic model but is clearly negative in a financial context. FinBERT
was fine-tuned on financial text (Malo et al., 2014 - Financial PhraseBank)
specifically to fix this. This is the standard sentiment model choice in
finance NLP research papers, so using it also strengthens your paper's
methodology section.

Model used: ProsusAI/finbert (on HuggingFace, free, ~440MB download, CPU-friendly)

Usage:
    python src/sentiment.py
"""

import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "ProsusAI/finbert"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
NEWS_ARCHIVE_PATH = os.path.join(RAW_DIR, "news_archive.csv")
SENTIMENT_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "news_with_sentiment.csv")

_tokenizer = None
_model = None


def load_model():
    """
    Loads FinBERT tokenizer + model once (lazily) and caches in module-level
    globals so repeated calls to score_headlines() don't reload the model
    every time (that would be slow -- model loading takes several seconds).
    """
    global _tokenizer, _model
    if _model is None:
        print(f"[model] Loading {MODEL_NAME} (first run downloads ~440MB, then it's cached locally)...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()  # inference mode, not training
        print("[model] Loaded.")
    return _tokenizer, _model


def score_headlines(headlines: list, batch_size: int = 16) -> pd.DataFrame:
    """
    Scores a list of headline strings with FinBERT.

    Returns a DataFrame with columns:
        title, label (positive/negative/neutral), score (confidence 0-1),
        sentiment_numeric (positive=+1, neutral=0, negative=-1) * confidence
        -> this signed numeric score is what we'll correlate with returns later.

    FinBERT's label order for ProsusAI/finbert is: [positive, negative, neutral]
    (index 0, 1, 2 respectively) -- confirmed from the model config.
    """
    tokenizer, model = load_model()
    results = []

    label_map = {0: "positive", 1: "negative", 2: "neutral"}
    signed_map = {"positive": 1, "negative": -1, "neutral": 0}

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        for headline, prob_row in zip(batch, probs):
            prob_row = prob_row.tolist()
            pred_idx = int(torch.argmax(torch.tensor(prob_row)))
            label = label_map[pred_idx]
            confidence = prob_row[pred_idx]
            signed_score = signed_map[label] * confidence

            results.append({
                "title": headline,
                "label": label,
                "confidence": confidence,
                "sentiment_score": signed_score,  # ranges from -1 (very negative) to +1 (very positive)
                "prob_positive": prob_row[0],
                "prob_negative": prob_row[1],
                "prob_neutral": prob_row[2],
            })

    return pd.DataFrame(results)


def score_news_archive():
    """
    Loads the news archive built by fetch_news.py, scores every headline,
    merges the sentiment columns back in, and saves to data/processed/.
    """
    if not os.path.exists(NEWS_ARCHIVE_PATH):
        raise FileNotFoundError(
            f"No news archive found at {NEWS_ARCHIVE_PATH}. Run fetch_news.py first."
        )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    news_df = pd.read_csv(NEWS_ARCHIVE_PATH, parse_dates=["datetime"])
    print(f"[sentiment] Scoring {len(news_df)} headlines...")

    sentiment_df = score_headlines(news_df["title"].tolist())

    # IMPORTANT: merge by POSITION (reset_index + concat), not by joining on the
    # `title` column. news_df is de-duplicated per-ticker (see fetch_news.py's
    # drop_duplicates(subset=["ticker","title"])), so the SAME headline text can
    # legitimately appear more than once across different tickers (e.g. a wire
    # headline like "Fed holds rates steady" tagged under both AAPL and MSFT).
    # Merging on title alone would then match one news_df row to multiple
    # sentiment_df rows (or vice versa), silently duplicating events and
    # inflating the dataset. score_headlines() preserves row order and produces
    # exactly one output row per input row, so a position-based concat is safe
    # and avoids this entirely.
    news_df = news_df.reset_index(drop=True)
    sentiment_df = sentiment_df.reset_index(drop=True)
    assert len(news_df) == len(sentiment_df), (
        f"Row count mismatch after scoring: {len(news_df)} headlines in, "
        f"{len(sentiment_df)} sentiment rows out. This should never happen -- "
        f"check score_headlines() for a bug if it does."
    )
    merged = pd.concat([news_df, sentiment_df.drop(columns=["title"])], axis=1)
    merged.to_csv(SENTIMENT_OUTPUT_PATH, index=False)
    print(f"[sentiment] Saved {len(merged)} scored headlines -> {SENTIMENT_OUTPUT_PATH}")

    print("\nSentiment label distribution:")
    print(merged["label"].value_counts())

    return merged


if __name__ == "__main__":
    score_news_archive()