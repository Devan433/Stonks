"""
RSS news scraper for Indian financial markets.

Fetches headlines from Economic Times, Moneycontrol, and Business Standard,
matches them against the stock universe, and stores in SQLite.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

import feedparser
import pytz
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from trading_assistant.config import DATA, MARKET, STOCKS
from trading_assistant.data.database import DatabaseManager

logger = logging.getLogger(__name__)
IST = pytz.timezone(MARKET.TIMEZONE)


# ── Stock-matching helpers ──────────────────────────────────────

def _build_match_patterns() -> Dict[re.Pattern, str]:
    """Compile regex patterns for each company name → ticker mapping.

    Patterns use word boundaries so 'SBI' matches 'SBI Q4 results'
    but not 'SBIG' or 'uSBI'.
    """
    patterns = {}
    for name, ticker in STOCKS.COMPANY_NAMES.items():
        pat = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        patterns[pat] = ticker
    return patterns


_PATTERNS = _build_match_patterns()


def match_headline_to_stocks(headline: str) -> List[str]:
    """Return a deduplicated list of tickers mentioned in *headline*.

    Uses word-boundary regex matching against STOCKS.COMPANY_NAMES.

    Args:
        headline: News headline text.

    Returns:
        List of matching tickers, e.g. ['RELIANCE.NS', 'SBIN.NS'].
    """
    matches: set = set()
    for pattern, ticker in _PATTERNS.items():
        if pattern.search(headline):
            matches.add(ticker)
    return sorted(matches)


# ── RSS feed parser ─────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True,
)
def _parse_feed(url: str) -> feedparser.FeedParserDict:
    """Parse an RSS feed URL with retry logic.

    Args:
        url: RSS feed URL.

    Returns:
        Parsed feedparser result.
    """
    result = feedparser.parse(url)
    if result.bozo and not result.entries:
        raise ConnectionError(f"Feed parse error for {url}: {result.bozo_exception}")
    return result


def _parse_published_time(entry) -> str:
    """Extract and normalise the published timestamp to ISO IST.

    Falls back to current time if the entry lacks a valid date.

    Args:
        entry: A feedparser entry object.

    Returns:
        ISO-formatted IST datetime string.
    """
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        utc_dt = datetime(*entry.published_parsed[:6], tzinfo=pytz.UTC)
        return utc_dt.astimezone(IST).isoformat()

    # Fallback: use current time
    return datetime.now(IST).isoformat()


# ── Public API ──────────────────────────────────────────────────

def fetch_news_from_feed(
    url: str,
    source: str,
) -> List[dict]:
    """Fetch and parse a single RSS feed.

    Each article is matched against the stock universe.
    Unmatched articles are still returned (with empty matched_symbols)
    for general-market sentiment.

    Args:
        url:    RSS feed URL.
        source: Human-readable source name (e.g. 'Economic Times').

    Returns:
        List of article dicts ready for DatabaseManager.store_news_articles().
    """
    try:
        feed = _parse_feed(url)
    except Exception as exc:
        logger.error("Failed to fetch %s (%s): %s", source, url, exc)
        return []

    articles: List[dict] = []
    for entry in feed.entries:
        headline = entry.get("title", "").strip()
        if not headline:
            continue

        matched = match_headline_to_stocks(headline)

        articles.append({
            "headline": headline,
            "published_at": _parse_published_time(entry),
            "source": source,
            "link": entry.get("link", ""),
            "matched_symbols": ",".join(matched),
        })

    logger.info("Parsed %d articles from %s (%d stock-matched)",
                len(articles), source,
                sum(1 for a in articles if a["matched_symbols"]))
    return articles


def fetch_all_news(store: bool = True) -> List[dict]:
    """Scrape all configured RSS feeds and optionally store results.

    Iterates over DATA.RSS_FEEDS, collects articles, deduplicates
    by headline+source via the database UNIQUE constraint.

    Args:
        store: Persist to SQLite via DatabaseManager.

    Returns:
        Combined list of all article dicts.
    """
    all_articles: List[dict] = []

    for feed_cfg in DATA.RSS_FEEDS:
        articles = fetch_news_from_feed(feed_cfg["url"], feed_cfg["source"])
        all_articles.extend(articles)

    if store and all_articles:
        with DatabaseManager() as db:
            db.store_news_articles(all_articles)

    logger.info("Total news fetched: %d articles from %d feeds",
                len(all_articles), len(DATA.RSS_FEEDS))
    return all_articles


def get_stock_news(
    symbol: str,
    hours: int = 24,
) -> List[dict]:
    """Convenience: pull recent news for a single stock from the database.

    Args:
        symbol: Ticker, e.g. 'RELIANCE.NS'.
        hours:  Look-back window in hours.

    Returns:
        List of article dicts (may be empty).
    """
    with DatabaseManager() as db:
        df = db.get_recent_news(hours=hours, symbol=symbol)
    return df.to_dict("records") if not df.empty else []
