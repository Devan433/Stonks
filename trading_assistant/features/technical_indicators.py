"""
Technical indicators computed on OHLCV price data.

Pure pandas/numpy implementations (no external TA library needed):
  RSI (14), MACD (12/26/9), Bollinger Bands (20/2), VWAP,
  EMA-9, EMA-21, ATR (14), and a volume-spike detector.

Returns a clean DataFrame ready for ML feature input — warmup
NaN rows are dropped automatically.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing).

    RSI = 100 - 100 / (1 + avg_gain / avg_loss)

    Args:
        close: Close price series.
        period: Look-back window (default 14).

    Returns:
        Series named 'rsi_{period}'.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's exponential moving average
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename(f"rsi_{period}")


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    Returns DataFrame with: macd_line, macd_signal, macd_histogram.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_hist,
    })


def compute_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands — lower, mid, upper, bandwidth, and %B.

    Returns DataFrame with: bb_lower, bb_mid, bb_upper, bb_bandwidth, bb_percent.
    """
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    pct_b = (close - lower) / (upper - lower)

    return pd.DataFrame({
        "bb_lower": lower,
        "bb_mid": sma,
        "bb_upper": upper,
        "bb_bandwidth": bandwidth,
        "bb_percent": pct_b,
    })


def compute_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume-Weighted Average Price.

    VWAP = cumsum(typical_price * volume) / cumsum(volume).
    Resets daily if the index contains date information.

    Returns:
        Series named 'vwap'.
    """
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).cumsum()
    cum_vol = volume.cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap.rename("vwap")


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average for a given period.

    Returns:
        Series named 'ema_{period}'.
    """
    return close.ewm(span=period, adjust=False).mean().rename(f"ema_{period}")


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range — measures volatility.

    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = EMA(TR, period)

    Returns:
        Series named 'atr_{period}'.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return atr.rename(f"atr_{period}")


def compute_volume_spike(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume spike detector.

    Flags bars where current volume exceeds 2× the 20-period SMA.
    Returns an integer Series (0 or 1) named 'volume_spike'.
    """
    vol_sma = volume.rolling(window=period).mean()
    spike = (volume > 2.0 * vol_sma).astype(int)
    return spike.rename("volume_spike")


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full indicator suite and return a clean DataFrame.

    Concatenates all individual indicator outputs with the original
    OHLCV columns, then adds Smart Money Concept features, and drops
    rows that contain NaN values caused by indicator warm-up periods.

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]
            and a DatetimeIndex.

    Returns:
        Enriched DataFrame with ~24 additional feature columns,
        NaN-free and ready for ML input.
    """
    if df.empty or len(df) < 30:
        logger.warning("DataFrame too short (%d rows) for indicators", len(df))
        return df

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for indicators: {missing}")

    logger.info("Computing indicators on %d rows", len(df))

    indicators = [
        compute_rsi(df["close"]),
        compute_macd(df["close"]),
        compute_bollinger_bands(df["close"]),
        compute_vwap(df["high"], df["low"], df["close"], df["volume"]),
        compute_ema(df["close"], period=9),
        compute_ema(df["close"], period=21),
        compute_atr(df["high"], df["low"], df["close"]),
        compute_volume_spike(df["volume"]),
    ]

    result = pd.concat([df] + indicators, axis=1)

    # Add Smart Money Concepts (FVG, Order Blocks, Liquidity Sweeps)
    from trading_assistant.features.smc_logic import compute_smc_features
    smc = compute_smc_features(result)
    # Only grab the new SMC columns to avoid duplicating OHLCV
    smc_cols = [c for c in smc.columns if c not in result.columns]
    if smc_cols:
        result = pd.concat([result, smc[smc_cols]], axis=1)

    before = len(result)
    result.dropna(inplace=True)
    after = len(result)

    logger.info("Indicators ready: %d → %d rows (dropped %d warmup)",
                before, after, before - after)
    return result


def get_indicator_summary(df: pd.DataFrame) -> dict:
    """Return the latest indicator values as a dict (for dashboard display).

    Args:
        df: DataFrame from compute_all_indicators().

    Returns:
        Dict with the latest value for each indicator.
    """
    if df.empty:
        return {}

    latest = df.iloc[-1]
    cols = [
        "rsi_14", "macd_line", "macd_signal", "macd_histogram",
        "bb_lower", "bb_mid", "bb_upper", "bb_bandwidth", "bb_percent",
        "vwap", "ema_9", "ema_21", "atr_14", "volume_spike",
    ]
    return {c: round(float(latest[c]), 4) for c in cols if c in latest.index}
