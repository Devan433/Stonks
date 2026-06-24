"""
Backtesting engine using the backtesting.py library.

Strategy: Buy on UP signal, sell on DOWN signal, hold on SIDEWAYS.
Constraints:
  • 0.1% brokerage per trade
  • 0.05% slippage (modelled via trade_on_close)
  • No trading in first 15 minutes (9:15–9:30 AM)
  • Max 20% of portfolio per position

Compares strategy return against Nifty 50 buy-and-hold benchmark.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import yfinance as yf

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from trading_assistant.config import MARKET, PATHS, STOCKS, TRADING

logger = logging.getLogger(__name__)
IST = pytz.timezone(MARKET.TIMEZONE)


# ── Signal-Based Strategy ───────────────────────────────────────

class SignalStrategy(Strategy):
    """Long-only strategy driven by ML signal column.

    Expects the input DataFrame to contain a 'signal' column with
    integer values: 2 = UP (buy), 0 = DOWN (sell), 1 = SIDEWAYS (hold).

    Position sizing: uses max MAX_POSITION_PCT of equity per trade.
    No-trade zone: 9:15–9:30 AM IST is skipped.
    """

    # Class-level parameters (set before Backtest.run)
    max_position_pct = TRADING.MAX_POSITION_PCT
    no_trade_hour = MARKET.NO_TRADE_END[0]
    no_trade_minute = MARKET.NO_TRADE_END[1]

    def init(self):
        """Pre-compute the signal indicator series."""
        self.signal = self.I(lambda x: x, self.data.signal, name="ML Signal")

    def next(self):
        """Execute trades based on the current signal value."""
        current_time = self.data.index[-1]

        # Skip the first 15 minutes (volatile open)
        if hasattr(current_time, "hour"):
            time_mins = current_time.hour * 60 + current_time.minute
            no_trade_mins = self.no_trade_hour * 60 + self.no_trade_minute
            if time_mins < no_trade_mins:
                return

        sig = self.signal[-1]

        if sig == 2 and not self.position:
            # UP signal → buy with position sizing
            self.buy(size=self.max_position_pct)

        elif sig == 0 and self.position:
            # DOWN signal → close position
            self.position.close()


# ── Backtest Runner ─────────────────────────────────────────────

def prepare_backtest_data(
    df: pd.DataFrame,
    signals: pd.Series,
) -> pd.DataFrame:
    """Merge OHLCV data with ML signals into backtesting.py format.

    backtesting.py requires columns: Open, High, Low, Close, Volume.
    We add 'signal' as an extra column.

    Args:
        df: OHLCV DataFrame with lowercase columns and DatetimeIndex.
        signals: Integer-encoded signal Series aligned with df index
                 (0=DOWN, 1=SIDEWAYS, 2=UP).

    Returns:
        DataFrame with capitalised OHLCV columns + 'signal'.
    """
    bt_df = df[["open", "high", "low", "close", "volume"]].copy()
    bt_df.columns = ["Open", "High", "Low", "Close", "Volume"]
    bt_df["signal"] = signals.reindex(bt_df.index).fillna(1).astype(int)

    # Ensure index is timezone-naive (backtesting.py requirement)
    if bt_df.index.tz is not None:
        bt_df.index = bt_df.index.tz_localize(None)

    return bt_df


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    cash: float = TRADING.INITIAL_CAPITAL,
    commission: float = TRADING.BROKERAGE_PCT,
    save_dir: Optional[Path] = None,
) -> Dict:
    """Execute the backtest and return performance metrics.

    Args:
        df: OHLCV DataFrame.
        signals: ML signal Series (0/1/2).
        cash: Starting capital.
        commission: Commission per trade (0.001 = 0.1%).
        save_dir: Directory to save the interactive HTML report.

    Returns:
        Dict with keys: metrics (dict), trades (DataFrame),
        equity_curve (DataFrame), stats (backtest Stats object).
    """
    bt_df = prepare_backtest_data(df, signals)

    bt = Backtest(
        bt_df,
        SignalStrategy,
        cash=cash,
        commission=commission,
        trade_on_close=True,   # model slippage
        exclusive_orders=True,
    )

    stats = bt.run()

    # Save interactive HTML report
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        report_path = save_dir / "backtest_report.html"
        bt.plot(filename=str(report_path), open_browser=False)
        logger.info("Backtest report saved to %s", report_path)

    # Extract key metrics
    metrics = _extract_metrics(stats)
    trades = stats._trades if hasattr(stats, "_trades") else pd.DataFrame()
    equity = stats._equity_curve if hasattr(stats, "_equity_curve") else pd.DataFrame()

    logger.info("Backtest complete — Return: %.2f%%, Sharpe: %.2f, Trades: %d",
                metrics["total_return_pct"], metrics["sharpe_ratio"],
                metrics["n_trades"])

    return {
        "metrics": metrics,
        "trades": trades,
        "equity_curve": equity,
        "stats": stats,
    }


def _extract_metrics(stats) -> Dict:
    """Pull key performance metrics from backtesting.py Stats object."""
    return {
        "total_return_pct": round(float(stats["Return [%]"]), 2),
        "sharpe_ratio": round(float(stats.get("Sharpe Ratio", 0) or 0), 2),
        "max_drawdown_pct": round(float(stats.get("Max. Drawdown [%]", 0) or 0), 2),
        "win_rate_pct": round(float(stats.get("Win Rate [%]", 0) or 0), 2),
        "n_trades": int(stats.get("# Trades", 0) or 0),
        "avg_trade_pct": round(float(stats.get("Avg. Trade [%]", 0) or 0), 2),
        "profit_factor": round(float(stats.get("Profit Factor", 0) or 0), 2),
        "exposure_pct": round(float(stats.get("Exposure Time [%]", 0) or 0), 2),
        "final_equity": round(float(stats.get("Equity Final [$]", 0) or 0), 2),
    }


# ── Benchmark Comparison ───────────────────────────────────────

def fetch_benchmark_return(
    start: str,
    end: str,
    ticker: str = STOCKS.BENCHMARK,
) -> float:
    """Calculate Nifty 50 buy-and-hold return over the given period.

    Args:
        start: Start date (YYYY-MM-DD).
        end:   End date (YYYY-MM-DD).
        ticker: Benchmark ticker (default ^NSEI).

    Returns:
        Percentage return (e.g. 12.5 for +12.5%).
    """
    try:
        data = yf.Ticker(ticker).history(start=start, end=end)
        if data.empty or len(data) < 2:
            return 0.0
        ret = (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100
        return round(ret, 2)
    except Exception as exc:
        logger.error("Benchmark fetch failed: %s", exc)
        return 0.0


def generate_backtest_report(
    bt_results: Dict,
    start_date: str,
    end_date: str,
    save_dir: Optional[Path] = None,
) -> Dict:
    """Create a full report comparing strategy vs benchmark.

    Args:
        bt_results: Output of run_backtest().
        start_date: Backtest period start (YYYY-MM-DD).
        end_date:   Backtest period end (YYYY-MM-DD).
        save_dir:   Where to save the report artefacts.

    Returns:
        Enriched metrics dict with benchmark comparison.
    """
    save_dir = save_dir or (PATHS.MODEL_DIR / "backtest_results")
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = bt_results["metrics"]
    benchmark_ret = fetch_benchmark_return(start_date, end_date)
    metrics["benchmark_return_pct"] = benchmark_ret
    metrics["alpha_pct"] = round(metrics["total_return_pct"] - benchmark_ret, 2)

    # Save metrics JSON
    with open(save_dir / "backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save trades CSV
    trades = bt_results.get("trades")
    if trades is not None and not trades.empty:
        trades.to_csv(save_dir / "trade_log.csv", index=False)

    logger.info("Strategy: %.2f%% | Benchmark: %.2f%% | Alpha: %.2f%%",
                metrics["total_return_pct"], benchmark_ret, metrics["alpha_pct"])

    return metrics
