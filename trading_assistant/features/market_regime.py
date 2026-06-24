"""
Market Regime Filter — crash protection system.

Determines the current market regime by analysing:
  1. India VIX (fear gauge) — spikes during panics.
  2. Nifty 50 trend — SMA crossover for bull/bear detection.

Regime states:
  • BULLISH   — VIX low, Nifty above MA → all signals allowed.
  • CAUTIOUS  — VIX elevated (>20) → reduce position sizes.
  • BEARISH   — Nifty below MA → block BUY signals.
  • PANIC     — VIX extreme (>25) → block ALL BUY signals immediately.
"""

import logging
from enum import Enum
from typing import Optional

import pandas as pd
import yfinance as yf

from trading_assistant.config import REGIME, STOCKS

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULLISH = "BULLISH"
    CAUTIOUS = "CAUTIOUS"
    BEARISH = "BEARISH"
    PANIC = "PANIC"


def fetch_vix() -> Optional[float]:
    """Fetch the latest India VIX value."""
    try:
        vix = yf.Ticker(STOCKS.VIX_TICKER)
        hist = vix.history(period="5d", interval="1d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.error("Failed to fetch India VIX: %s", exc)
        return None


def fetch_nifty_trend() -> Optional[bool]:
    """Check if Nifty 50 is trading above its 50-day SMA.

    Returns:
        True if Nifty is ABOVE the SMA (bullish trend).
        False if Nifty is BELOW the SMA (bearish trend).
        None if data fetch failed.
    """
    try:
        nifty = yf.Ticker(STOCKS.BENCHMARK)
        hist = nifty.history(period="6mo", interval="1d")
        if hist.empty or len(hist) < REGIME.NIFTY_MA_PERIOD:
            return None

        sma = hist["Close"].rolling(window=REGIME.NIFTY_MA_PERIOD).mean()
        latest_close = hist["Close"].iloc[-1]
        latest_sma = sma.iloc[-1]

        is_bullish = latest_close > latest_sma
        logger.info(
            "Nifty 50: %.2f | SMA(%d): %.2f | Trend: %s",
            latest_close, REGIME.NIFTY_MA_PERIOD, latest_sma,
            "BULLISH" if is_bullish else "BEARISH",
        )
        return is_bullish
    except Exception as exc:
        logger.error("Failed to fetch Nifty 50 trend: %s", exc)
        return None


def detect_regime() -> MarketRegime:
    """Determine the current market regime.

    Decision logic (evaluated top-down, first match wins):
      1. VIX > 25       → PANIC  (extreme fear, block everything)
      2. VIX > 20       → CAUTIOUS (elevated fear, reduce sizes)
      3. Nifty < SMA    → BEARISH (downtrend, block BUY)
      4. Otherwise       → BULLISH (all clear)
    """
    vix = fetch_vix()
    nifty_bullish = fetch_nifty_trend()

    # Default to CAUTIOUS if data is unavailable (fail-safe)
    if vix is None and nifty_bullish is None:
        logger.warning("Could not fetch VIX or Nifty data — defaulting to CAUTIOUS")
        return MarketRegime.CAUTIOUS

    # Check VIX panic levels first (highest priority)
    if vix is not None:
        if vix > REGIME.VIX_PANIC:
            logger.warning("🚨 PANIC REGIME — VIX at %.2f (> %.1f threshold)",
                           vix, REGIME.VIX_PANIC)
            return MarketRegime.PANIC

        if vix > REGIME.VIX_CAUTION:
            logger.warning("⚠️ CAUTIOUS REGIME — VIX at %.2f (> %.1f threshold)",
                           vix, REGIME.VIX_CAUTION)
            return MarketRegime.CAUTIOUS

    # Check Nifty trend
    if nifty_bullish is not None and not nifty_bullish:
        logger.warning("📉 BEARISH REGIME — Nifty 50 below %d-period SMA",
                       REGIME.NIFTY_MA_PERIOD)
        return MarketRegime.BEARISH

    logger.info("✅ BULLISH REGIME — VIX: %s, Nifty trend: UP",
                f"{vix:.2f}" if vix else "N/A")
    return MarketRegime.BULLISH


def should_allow_buy(regime: MarketRegime) -> bool:
    """Return True if BUY signals are allowed under the current regime."""
    return regime == MarketRegime.BULLISH or regime == MarketRegime.CAUTIOUS


def get_position_scale(regime: MarketRegime) -> float:
    """Return a position sizing multiplier based on regime.

    BULLISH  → 1.0  (full position)
    CAUTIOUS → 0.5  (half position)
    BEARISH  → 0.0  (no buys)
    PANIC    → 0.0  (no buys)
    """
    scale_map = {
        MarketRegime.BULLISH: 1.0,
        MarketRegime.CAUTIOUS: 0.5,
        MarketRegime.BEARISH: 0.0,
        MarketRegime.PANIC: 0.0,
    }
    return scale_map.get(regime, 0.0)
