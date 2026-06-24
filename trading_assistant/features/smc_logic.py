"""
Smart Money Concepts (SMC) — institutional footprint detection.

Detects key structures left by large institutional players:
  • Fair Value Gaps (FVG)  — price imbalances from aggressive volume.
  • Order Blocks (OB)      — zones where institutions placed massive orders.
  • Liquidity Sweeps       — stop-loss hunts before a reversal.

These features are appended as columns to the OHLCV DataFrame and fed
into the ML model alongside standard technical indicators.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Fair Value Gaps ─────────────────────────────────────────────

def detect_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Fair Value Gaps (FVG) in the price action.

    A Bullish FVG occurs when:
        candle[i-2].high < candle[i].low  (a gap between the bodies)

    A Bearish FVG occurs when:
        candle[i-2].low > candle[i].high

    Returns DataFrame with columns:
        fvg_bullish (1/0), fvg_bearish (1/0), fvg_size (gap size as % of close)
    """
    high = np.asarray(df["high"].values, dtype=np.float64)
    low = np.asarray(df["low"].values, dtype=np.float64)
    close = np.asarray(df["close"].values, dtype=np.float64)
    n = len(df)

    bullish = np.zeros(n, dtype=int)
    bearish = np.zeros(n, dtype=int)
    gap_size = np.zeros(n, dtype=float)

    for i in range(2, n):
        # Bullish FVG: candle 2 bars ago high < current candle low
        if high[i - 2] < low[i]:
            bullish[i] = 1
            gap_size[i] = (low[i] - high[i - 2]) / close[i] * 100

        # Bearish FVG: candle 2 bars ago low > current candle high
        if low[i - 2] > high[i]:
            bearish[i] = 1
            gap_size[i] = (low[i - 2] - high[i]) / close[i] * 100

    return pd.DataFrame({
        "fvg_bullish": bullish,
        "fvg_bearish": bearish,
        "fvg_size": gap_size,
    }, index=df.index)


# ── Order Blocks ────────────────────────────────────────────────

def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = 10,
    volume_threshold: float = 1.5,
) -> pd.DataFrame:
    """Detect Order Blocks — zones where institutional volume kicked in.

    A Bullish Order Block is the last bearish (red) candle before a strong
    bullish move, accompanied by above-average volume.

    A Bearish Order Block is the last bullish (green) candle before a strong
    bearish move, accompanied by above-average volume.

    Args:
        df: OHLCV DataFrame.
        lookback: Number of bars to check for the subsequent move.
        volume_threshold: Volume must be > threshold * SMA(20) to qualify.

    Returns:
        DataFrame with columns:
            ob_bullish (1/0), ob_bearish (1/0),
            ob_strength (magnitude of subsequent move as %).
    """
    open_ = np.asarray(df["open"].values, dtype=np.float64)
    close = np.asarray(df["close"].values, dtype=np.float64)
    high = np.asarray(df["high"].values, dtype=np.float64)
    low = np.asarray(df["low"].values, dtype=np.float64)
    volume = np.asarray(df["volume"].values, dtype=np.float64)
    n = len(df)

    vol_sma = np.asarray(pd.Series(volume.tolist()).rolling(20).mean().fillna(float(np.mean(volume))))

    bull_ob = np.zeros(n, dtype=int)
    bear_ob = np.zeros(n, dtype=int)
    strength = np.zeros(n, dtype=float)

    for i in range(1, n - lookback):
        is_high_volume = volume[i] > volume_threshold * vol_sma[i]
        if not is_high_volume:
            continue

        is_bearish_candle = close[i] < open_[i]
        is_bullish_candle = close[i] > open_[i]

        # Check for strong move in the following bars
        future_high = float(np.max(high[i + 1: i + 1 + lookback]))
        future_low = float(np.min(low[i + 1: i + 1 + lookback]))

        move_up = (future_high - close[i]) / close[i] * 100
        move_down = (close[i] - future_low) / close[i] * 100

        # Bullish OB: bearish candle followed by strong upward move (>1%)
        if is_bearish_candle and move_up > 1.0:
            bull_ob[i] = 1
            strength[i] = move_up

        # Bearish OB: bullish candle followed by strong downward move (>1%)
        if is_bullish_candle and move_down > 1.0:
            bear_ob[i] = 1
            strength[i] = move_down

    return pd.DataFrame({
        "ob_bullish": bull_ob,
        "ob_bearish": bear_ob,
        "ob_strength": strength,
    }, index=df.index)


# ── Liquidity Sweeps ───────────────────────────────────────────

def detect_liquidity_sweeps(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """Detect Liquidity Sweeps — stop-loss hunts by institutional players.

    A Bullish Sweep occurs when price dips below a recent swing low
    (hunting stop losses) and then reverses strongly upward.

    A Bearish Sweep occurs when price spikes above a recent swing high
    (hunting stop losses) and then reverses strongly downward.

    Args:
        df: OHLCV DataFrame.
        lookback: Window for finding recent swing highs/lows.

    Returns:
        DataFrame with columns:
            sweep_bullish (1/0), sweep_bearish (1/0), sweep_depth (%).
    """
    high = np.asarray(df["high"].values, dtype=np.float64)
    low = np.asarray(df["low"].values, dtype=np.float64)
    close = np.asarray(df["close"].values, dtype=np.float64)
    open_ = np.asarray(df["open"].values, dtype=np.float64)
    n = len(df)

    bull_sweep = np.zeros(n, dtype=int)
    bear_sweep = np.zeros(n, dtype=int)
    depth = np.zeros(n, dtype=float)

    for i in range(lookback, n):
        # Find the recent swing low / high
        recent_low = float(np.min(low[i - lookback: i]))
        recent_high = float(np.max(high[i - lookback: i]))

        # Bullish sweep: price dips below recent low then closes back up
        if low[i] < recent_low and close[i] > open_[i]:
            bull_sweep[i] = 1
            depth[i] = (recent_low - low[i]) / close[i] * 100

        # Bearish sweep: price spikes above recent high then closes down
        if high[i] > recent_high and close[i] < open_[i]:
            bear_sweep[i] = 1
            depth[i] = (high[i] - recent_high) / close[i] * 100

    return pd.DataFrame({
        "sweep_bullish": bull_sweep,
        "sweep_bearish": bear_sweep,
        "sweep_depth": depth,
    }, index=df.index)


# ── Combined SMC Features ──────────────────────────────────────

def compute_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all Smart Money Concept features and return combined DataFrame.

    Appends 9 new columns to the input DataFrame:
      fvg_bullish, fvg_bearish, fvg_size,
      ob_bullish, ob_bearish, ob_strength,
      sweep_bullish, sweep_bearish, sweep_depth.

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume].

    Returns:
        Original DataFrame enriched with SMC feature columns.
    """
    if df.empty or len(df) < 30:
        logger.warning("DataFrame too short (%d rows) for SMC analysis", len(df))
        return df

    logger.info("Computing SMC features on %d rows", len(df))

    fvg = detect_fair_value_gaps(df)
    ob = detect_order_blocks(df)
    sweeps = detect_liquidity_sweeps(df)

    result = pd.concat([df, fvg, ob, sweeps], axis=1)

    n_fvg = fvg["fvg_bullish"].sum() + fvg["fvg_bearish"].sum()
    n_ob = ob["ob_bullish"].sum() + ob["ob_bearish"].sum()
    n_sweep = sweeps["sweep_bullish"].sum() + sweeps["sweep_bearish"].sum()

    logger.info(
        "SMC features: %d FVGs, %d Order Blocks, %d Liquidity Sweeps detected",
        n_fvg, n_ob, n_sweep,
    )
    return result
