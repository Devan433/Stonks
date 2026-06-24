"""
SQLite database handler for the Trading Assistant.

Provides thread-safe storage for price data, news articles,
sentiment scores, and trade signals with proper indexing.
Uses WAL journal mode for better concurrent read/write performance.
"""

import logging
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional

import pandas as pd

from trading_assistant.config import PATHS

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Thread-safe SQLite manager for trading data.

    Uses INSERT OR REPLACE for upsert semantics on price data,
    and INSERT OR IGNORE for news deduplication.
    """

    _lock = threading.Lock()

    def __init__(self, db_path: Optional[Path] = None):
        """Open (or create) the database and ensure all tables exist.

        Args:
            db_path: Override path; defaults to PATHS.DB_PATH.
        """
        self.db_path = db_path or PATHS.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=30,
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()
        logger.info("Database initialised at %s", self.db_path)

    # ── Schema ──────────────────────────────────────────────────

    def _create_tables(self) -> None:
        """Create tables and indexes if they don't already exist."""
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT    NOT NULL,
                symbol    TEXT    NOT NULL,
                interval  TEXT    NOT NULL,
                open      REAL   NOT NULL,
                high      REAL   NOT NULL,
                low       REAL   NOT NULL,
                close     REAL   NOT NULL,
                volume    INTEGER NOT NULL,
                UNIQUE(timestamp, symbol, interval)
            )""")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                headline        TEXT NOT NULL,
                published_at    TEXT,
                source          TEXT,
                link            TEXT,
                matched_symbols TEXT,
                fetched_at      TEXT DEFAULT (datetime('now','localtime')),
                UNIQUE(headline, source)
            )""")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_scores (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id       INTEGER REFERENCES news_articles(id),
                symbol           TEXT NOT NULL,
                sentiment_label  TEXT NOT NULL,
                sentiment_score  REAL NOT NULL,
                timestamp_window TEXT NOT NULL,
                created_at       TEXT DEFAULT (datetime('now','localtime'))
            )""")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trade_signals (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT NOT NULL,
                symbol            TEXT NOT NULL,
                signal            TEXT NOT NULL,
                confidence        REAL,
                model_name        TEXT,
                features_snapshot TEXT,
                created_at        TEXT DEFAULT (datetime('now','localtime'))
            )""")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name   TEXT NOT NULL,
                metric_name  TEXT NOT NULL,
                metric_value REAL NOT NULL,
                fold_number  INTEGER,
                trained_at   TEXT DEFAULT (datetime('now','localtime'))
            )""")

        # Indexes for fast look-ups
        for stmt in (
            "CREATE INDEX IF NOT EXISTS idx_price_sym_ts    ON price_data(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_price_interval  ON price_data(interval, symbol)",
            "CREATE INDEX IF NOT EXISTS idx_news_pub        ON news_articles(published_at)",
            "CREATE INDEX IF NOT EXISTS idx_sent_sym_win    ON sentiment_scores(symbol, timestamp_window)",
            "CREATE INDEX IF NOT EXISTS idx_sig_sym_ts      ON trade_signals(symbol, timestamp)",
        ):
            cur.execute(stmt)

        self.conn.commit()

    # ── Price Data ──────────────────────────────────────────────

    def store_price_data(self, df: pd.DataFrame) -> int:
        """Upsert OHLCV rows from *df* (must have symbol & interval cols).

        Args:
            df: DataFrame with DatetimeIndex named 'timestamp' and columns
                [open, high, low, close, volume, symbol, interval].

        Returns:
            Number of rows written.
        """
        if df.empty:
            return 0

        rows = [
            (ts.isoformat(), r["symbol"], r["interval"],
             float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), int(r["volume"]))
            for ts, r in df.iterrows()
        ]

        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO price_data "
                "(timestamp,symbol,interval,open,high,low,close,volume) "
                "VALUES (?,?,?,?,?,?,?,?)", rows,
            )
            self.conn.commit()

        logger.debug("Stored %d price rows", len(rows))
        return len(rows)

    def get_price_data(
        self,
        symbol: str,
        interval: str = "15m",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame for *symbol*, ordered oldest-first.

        Args:
            symbol:   Ticker, e.g. 'RELIANCE.NS'.
            interval: '1m', '15m', or '1d'.
            start:    ISO lower bound (inclusive).
            end:      ISO upper bound (inclusive).
            limit:    Cap on most-recent rows returned.
        """
        query = ("SELECT timestamp,open,high,low,close,volume "
                 "FROM price_data WHERE symbol=? AND interval=?")
        params: list = [symbol, interval]

        if start:
            query += " AND timestamp>=?"
            params.append(start)
        if end:
            query += " AND timestamp<=?"
            params.append(end)

        query += " ORDER BY timestamp DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._lock:
            df = pd.read_sql_query(query, self.conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
        return df

    # ── News ────────────────────────────────────────────────────

    def store_news_articles(self, articles: List[dict]) -> int:
        """Insert articles, silently skipping duplicates (headline+source).

        Each dict must contain: headline, published_at, source, link,
        matched_symbols (comma-separated ticker string).
        """
        if not articles:
            return 0

        with self._lock:
            self.conn.executemany(
                "INSERT OR IGNORE INTO news_articles "
                "(headline,published_at,source,link,matched_symbols) "
                "VALUES (?,?,?,?,?)",
                [(a["headline"], a.get("published_at", ""),
                  a.get("source", ""), a.get("link", ""),
                  a.get("matched_symbols", "")) for a in articles],
            )
            self.conn.commit()
            count = self.conn.total_changes  # approximate

        logger.debug("Stored news batch (%d articles offered)", len(articles))
        return count

    def get_recent_news(
        self, hours: int = 24, symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return news from the last *hours* hours, optionally for *symbol*."""
        query = ("SELECT * FROM news_articles "
                 "WHERE fetched_at >= datetime('now','localtime',?)")
        params: list = [f"-{hours} hours"]

        if symbol:
            query += " AND matched_symbols LIKE ?"
            params.append(f"%{symbol}%")

        query += " ORDER BY published_at DESC"
        with self._lock:
            return pd.read_sql_query(query, self.conn, params=params)

    # ── Sentiment ───────────────────────────────────────────────

    def store_sentiment_scores(self, scores: List[dict]) -> int:
        """Bulk-insert sentiment scores.

        Each dict: article_id, symbol, sentiment_label,
        sentiment_score (-1..+1), timestamp_window (ISO).
        """
        if not scores:
            return 0
        rows = [(s.get("article_id"), s["symbol"], s["sentiment_label"],
                 s["sentiment_score"], s["timestamp_window"]) for s in scores]
        with self._lock:
            self.conn.executemany(
                "INSERT INTO sentiment_scores "
                "(article_id,symbol,sentiment_label,sentiment_score,timestamp_window) "
                "VALUES (?,?,?,?,?)", rows,
            )
            self.conn.commit()
        return len(rows)

    def get_sentiment(
        self, symbol: str, window: Optional[str] = None,
    ) -> pd.DataFrame:
        """Aggregate sentiment for *symbol*, optionally in a single *window*."""
        query = ("SELECT timestamp_window, AVG(sentiment_score) AS avg_sentiment, "
                 "COUNT(*) AS article_count FROM sentiment_scores WHERE symbol=?")
        params: list = [symbol]
        if window:
            query += " AND timestamp_window=?"
            params.append(window)
        query += " GROUP BY timestamp_window ORDER BY timestamp_window"
        with self._lock:
            return pd.read_sql_query(query, self.conn, params=params)

    # ── Trade Signals ───────────────────────────────────────────

    def store_trade_signal(self, signal: dict) -> None:
        """Persist a single trade signal dict."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO trade_signals "
                "(timestamp,symbol,signal,confidence,model_name,features_snapshot) "
                "VALUES (?,?,?,?,?,?)",
                (signal["timestamp"], signal["symbol"], signal["signal"],
                 signal.get("confidence"), signal.get("model_name"),
                 signal.get("features_snapshot")),
            )
            self.conn.commit()

    def get_latest_signals(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Return the most recent signal per stock."""
        query = ("SELECT symbol, signal, confidence, model_name, timestamp "
                 "FROM trade_signals WHERE id IN "
                 "(SELECT MAX(id) FROM trade_signals GROUP BY symbol)")
        params: list = []
        if symbol:
            query = ("SELECT symbol, signal, confidence, model_name, timestamp "
                     "FROM trade_signals WHERE symbol=? "
                     "ORDER BY id DESC LIMIT 1")
            params = [symbol]
        with self._lock:
            return pd.read_sql_query(query, self.conn, params=params)

    # ── Lifecycle ───────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
