"""
Financial sentiment analysis for news headlines.

Primary:  FinBERT (ProsusAI/finbert) — state-of-the-art for financial text.
Fallback: TextBlob — lightweight, no GPU required.

Both return a normalised score in [-1, +1] and a label
(positive / negative / neutral).
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pytz

from trading_assistant.config import MARKET
from trading_assistant.data.database import DatabaseManager

logger = logging.getLogger(__name__)
IST = pytz.timezone(MARKET.TIMEZONE)


# ── Abstract base ───────────────────────────────────────────────

class BaseSentimentAnalyzer(ABC):
    """Interface for pluggable sentiment backends."""

    @abstractmethod
    def analyze(self, text: str) -> Tuple[str, float]:
        """Score a single headline.

        Returns:
            (label, score) where label ∈ {positive, negative, neutral}
            and score ∈ [-1.0, +1.0].
        """

    def analyze_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Score multiple headlines.  Override for batch-optimised backends."""
        return [self.analyze(t) for t in texts]


# ── FinBERT analyser ────────────────────────────────────────────

class FinBERTAnalyzer(BaseSentimentAnalyzer):
    """HuggingFace FinBERT (ProsusAI/finbert) sentiment scorer.

    The model is loaded lazily on first call and cached for reuse.
    Runs on CUDA if available (RTX 4050 detected → GPU accelerated).
    """

    def __init__(self):
        self._pipeline = None

    def _load_model(self):
        """Lazy-load the FinBERT pipeline (downloads on first use)."""
        if self._pipeline is not None:
            return

        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        backend = "GPU (CUDA)" if device == 0 else "CPU"
        logger.info("Loading FinBERT on %s …", backend)

        self._pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=device,
            top_k=None,           # return all 3 class scores
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded successfully")

    def analyze(self, text: str) -> Tuple[str, float]:
        """Score a single headline using FinBERT.

        Converts FinBERT's 3-class probabilities into a single
        scalar: score = P(positive) - P(negative),  ∈ [-1, +1].
        """
        self._load_model()

        try:
            results = self._pipeline(text)[0]  # list of {label, score} dicts
            scores = {r["label"].lower(): r["score"] for r in results}

            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            composite = round(pos - neg, 4)

            # Dominant label
            label = max(scores, key=scores.get)
            return label, composite

        except Exception as exc:
            logger.warning("FinBERT error for '%s…': %s", text[:50], exc)
            return "neutral", 0.0

    def analyze_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Batch-score headlines — uses HF pipeline batching internally."""
        self._load_model()
        results: List[Tuple[str, float]] = []

        try:
            outputs = self._pipeline(texts, batch_size=16)
            for output in outputs:
                scores = {r["label"].lower(): r["score"] for r in output}
                pos = scores.get("positive", 0.0)
                neg = scores.get("negative", 0.0)
                label = max(scores, key=scores.get)
                results.append((label, round(pos - neg, 4)))
        except Exception as exc:
            logger.error("FinBERT batch error: %s", exc)
            results = [("neutral", 0.0)] * len(texts)

        return results


# ── TextBlob fallback ───────────────────────────────────────────

class TextBlobAnalyzer(BaseSentimentAnalyzer):
    """Lightweight sentiment via TextBlob polarity.

    Polarity ∈ [-1, +1] maps directly to our score convention.
    No GPU or large model download required.
    """

    def analyze(self, text: str) -> Tuple[str, float]:
        """Score a headline using TextBlob polarity."""
        from textblob import TextBlob

        try:
            polarity = TextBlob(text).sentiment.polarity
            score = round(polarity, 4)

            if score > 0.05:
                label = "positive"
            elif score < -0.05:
                label = "negative"
            else:
                label = "neutral"

            return label, score
        except Exception as exc:
            logger.warning("TextBlob error: %s", exc)
            return "neutral", 0.0


# ── Factory ─────────────────────────────────────────────────────

def get_analyzer(backend: str = "finbert") -> BaseSentimentAnalyzer:
    """Return a sentiment analyser instance.

    Args:
        backend: 'finbert' (default, GPU-accelerated) or 'textblob' (fast CPU).
    """
    if backend == "finbert":
        return FinBERTAnalyzer()
    return TextBlobAnalyzer()


# ── Aggregation helpers ─────────────────────────────────────────

def score_articles(
    articles: List[dict],
    analyzer: Optional[BaseSentimentAnalyzer] = None,
) -> List[dict]:
    """Score a batch of article dicts and return sentiment records.

    Each input dict must have 'headline' and 'matched_symbols' keys.
    Returns a list of sentiment-score dicts ready for DB insertion.
    """
    analyzer = analyzer or get_analyzer()
    headlines = [a["headline"] for a in articles]
    sentiments = analyzer.analyze_batch(headlines)

    records: List[dict] = []
    now_window = _current_window()

    for article, (label, score) in zip(articles, sentiments):
        symbols = article.get("matched_symbols", "")
        if not symbols:
            continue  # Skip articles not matched to any stock

        for sym in symbols.split(","):
            sym = sym.strip()
            if sym:
                records.append({
                    "article_id": article.get("id"),
                    "symbol": sym,
                    "sentiment_label": label,
                    "sentiment_score": score,
                    "timestamp_window": now_window,
                })

    return records


def aggregate_sentiment(
    symbol: str,
    window: Optional[str] = None,
) -> float:
    """Return the average sentiment score for *symbol* in a time window.

    Args:
        symbol: Ticker, e.g. 'RELIANCE.NS'.
        window: ISO time-window string.  None → latest available window.

    Returns:
        Score in [-1, +1].  Returns 0.0 if no data.
    """
    with DatabaseManager() as db:
        df = db.get_sentiment(symbol, window=window)

    if df.empty:
        return 0.0

    return round(float(df["avg_sentiment"].iloc[-1]), 4)


def _current_window() -> str:
    """Return the current 15-minute window as an ISO string.

    E.g. if now is 10:37 IST → '2025-01-15T10:30:00+05:30'.
    """
    now = datetime.now(IST)
    floored_min = (now.minute // 15) * 15
    window = now.replace(minute=floored_min, second=0, microsecond=0)
    return window.isoformat()
