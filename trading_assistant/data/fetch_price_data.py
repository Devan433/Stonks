"""
Fetch NSE stock price data via yfinance.

Handles two tiers of history:
  • Daily OHLCV  — up to 4 years (no yfinance cap on daily).
  • 15-min OHLCV — up to 60 days  (yfinance cap for intraday).

All timestamps are converted to IST.  Market-hours filtering is
applied to intraday data (9:15 AM – 3:30 PM).  Every yfinance call
is wrapped in tenacity retry logic with exponential back-off.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import pytz
import yfinance as yf
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from trading_assistant.config import DATA, MARKET, STOCKS

logger = logging.getLogger(__name__)
IST = pytz.timezone(MARKET.TIMEZONE)


# ── Market-hours helper ─────────────────────────────────────────

def is_market_open() -> bool:
    """Return True when the current IST time falls within NSE trading hours.

    Checks both day-of-week (Mon–Fri) and clock time (9:15–15:30 IST).
    Does **not** account for exchange holidays — add a holiday calendar
    if precision matters for scheduling.
    """
    now = datetime.now(IST)
    if now.weekday() not in MARKET.TRADING_DAYS:
        return False

    open_mins = MARKET.MARKET_OPEN[0] * 60 + MARKET.MARKET_OPEN[1]
    close_mins = MARKET.MARKET_CLOSE[0] * 60 + MARKET.MARKET_CLOSE[1]
    now_mins = now.hour * 60 + now.minute

    return open_mins <= now_mins <= close_mins


# ── Low-level yfinance wrapper (with retry) ─────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True,
)
def _fetch_yfinance(
    ticker: str,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Call yfinance with automatic retries on network errors.

    Args:
        ticker:   NSE symbol, e.g. 'RELIANCE.NS'.
        period:   yfinance period string ('4y', '60d', …).
        start/end: date strings 'YYYY-MM-DD' (used when period is None).
        interval: candle size — '1m','5m','15m','1h','1d'.

    Returns:
        Cleaned OHLCV DataFrame indexed by IST-aware timestamps.

    Raises:
        ValueError: if yfinance returns an empty frame.
    """
    stock = yf.Ticker(ticker)
    df = (stock.history(period=period, interval=interval)
          if period
          else stock.history(start=start, end=end, interval=interval))

    if df.empty:
        raise ValueError(f"No data for {ticker} ({interval})")

    # Standardise column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep]

    # Ensure IST-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(IST)
    else:
        df.index = df.index.tz_convert(IST)
    df.index.name = "timestamp"

    logger.info("Fetched %d rows for %s (%s)", len(df), ticker, interval)
    return df


# ── Market-hours filter ──────────────────────────────────────────

def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose timestamp falls within 9:15–15:30 IST."""
    if df.empty:
        return df

    open_m = MARKET.MARKET_OPEN[0] * 60 + MARKET.MARKET_OPEN[1]
    close_m = MARKET.MARKET_CLOSE[0] * 60 + MARKET.MARKET_CLOSE[1]
    mins = df.index.hour * 60 + df.index.minute
    mask = (mins >= open_m) & (mins <= close_m)

    filtered = df.loc[mask]
    logger.debug("Market-hours filter: %d → %d rows", len(df), len(filtered))
    return filtered


# ── Public fetchers ──────────────────────────────────────────────

def fetch_historical_daily(
    ticker: str,
    years: int = DATA.HISTORICAL_YEARS,
) -> pd.DataFrame:
    """Download long-term daily OHLCV (up to *years* years).

    Adds 'symbol' and 'interval' columns for database storage.
    Returns an empty DataFrame on failure (logged, not raised).
    """
    logger.info("Fetching %d yr daily data for %s", years, ticker)
    try:
        df = _fetch_yfinance(ticker, period=f"{years}y", interval="1d")
        df["symbol"] = ticker
        df["interval"] = "1d"
        return df
    except Exception as exc:
        logger.error("Daily fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_intraday(
    ticker: str,
    interval: str = "15m",
    days: int = DATA.INTRADAY_DAYS,
) -> pd.DataFrame:
    """Download intraday OHLCV, clamped to yfinance look-back limits.

    Market-hours filter is applied automatically.  Returns an empty
    DataFrame on failure.
    """
    limit = DATA.MAX_LOOKBACK.get(interval, 60)
    days = min(days, limit)
    logger.info("Fetching %dd of %s data for %s", days, interval, ticker)

    try:
        df = _fetch_yfinance(ticker, period=f"{days}d", interval=interval)
        df = _filter_market_hours(df)
        df["symbol"] = ticker
        df["interval"] = interval
        return df
    except Exception as exc:
        logger.error("Intraday fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


# ── Bulk operations ──────────────────────────────────────────────

def fetch_all_historical(
    tickers: Optional[List[str]] = None,
    store: bool = True,
) -> Dict[str, dict]:
    """Fetch daily + 15-min history for every stock in the universe.

    Rate-limits requests by DATA.FETCH_DELAY seconds between symbols.

    Args:
        tickers: Override list; defaults to STOCKS.TICKERS.
        store:   Persist to SQLite via DatabaseManager.

    Returns:
        {ticker: {'daily': DataFrame, 'intraday': DataFrame}}
    """
    # Late import to avoid circular dependency at module level
    from trading_assistant.data.database import DatabaseManager

    tickers = list(tickers or STOCKS.TICKERS)
    results: Dict[str, dict] = {}
    db = DatabaseManager() if store else None

    try:
        for idx, ticker in enumerate(tickers, 1):
            logger.info("[%d/%d] %s", idx, len(tickers), ticker)

            daily = fetch_historical_daily(ticker)
            time.sleep(DATA.FETCH_DELAY)
            intraday = fetch_intraday(ticker, interval="15m")

            results[ticker] = {"daily": daily, "intraday": intraday}

            if db:
                if not daily.empty:
                    db.store_price_data(daily)
                if not intraday.empty:
                    db.store_price_data(intraday)

            if idx < len(tickers):
                time.sleep(DATA.FETCH_DELAY)

        logger.info("Historical fetch complete for %d stocks", len(tickers))
    finally:
        if db:
            db.close()

    return results


def refresh_live_data(
    tickers: Optional[List[str]] = None,
    store: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Refresh the latest 1-day of 15-min data for all stocks.

    Designed to be called every 5 minutes during market hours.
    Skips silently if the market is closed.

    Returns:
        {ticker: DataFrame} for successfully refreshed stocks.
    """
    if not is_market_open():
        logger.info("Market closed — skipping live refresh")
        return {}

    from trading_assistant.data.database import DatabaseManager

    tickers = list(tickers or STOCKS.TICKERS)
    results: Dict[str, pd.DataFrame] = {}
    db = DatabaseManager() if store else None

    try:
        for idx, ticker in enumerate(tickers, 1):
            try:
                df = fetch_intraday(ticker, interval="15m", days=1)
                if not df.empty:
                    results[ticker] = df
                    if db:
                        db.store_price_data(df)
            except Exception as exc:
                logger.warning("Refresh failed for %s: %s", ticker, exc)

            if idx < len(tickers):
                time.sleep(DATA.FETCH_DELAY)
    finally:
        if db:
            db.close()

    logger.info("Live refresh: %d/%d stocks OK", len(results), len(tickers))
    return results
