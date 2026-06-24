"""
Telegram Bot — two-way interactive notifications and chart generation.

Features:
  • Push notifications for BUY/SELL signals with full trade details.
  • /chart TICKER — generates and sends a candlestick chart image.
  • /status        — shows current market regime and active signals.
  • /help          — lists all available commands.

Uses python-telegram-bot for async two-way messaging and
matplotlib/mplfinance for chart generation.
"""

import io
import json
import logging
import threading
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytz

from trading_assistant.config import MARKET, PATHS, STOCKS

logger = logging.getLogger(__name__)

import os
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
IST = pytz.timezone(MARKET.TIMEZONE)


# ── Basic Message Sending ───────────────────────────────────────

def send_telegram_message(text: str) -> bool:
    """Send a markdown-formatted message to the configured Telegram chat."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = json.dumps({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers)

    try:
        with urllib.request.urlopen(req) as response:
            return response.status == 200
    except urllib.error.URLError as e:
        logger.error("Telegram notification failed: %s", e)
        return False


def send_telegram_photo(photo_bytes: bytes, caption: str = "") -> bool:
    """Send a photo (PNG bytes) to the configured Telegram chat."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"

    # Build multipart form data manually
    boundary = "----TradingAssistantBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
        f"{TELEGRAM_CHAT_ID}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="caption"\r\n\r\n'
        f"{caption}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="parse_mode"\r\n\r\n'
        f"Markdown\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="photo"; filename="chart.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8")

    body += photo_bytes
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    try:
        with urllib.request.urlopen(req) as response:
            return response.status == 200
    except urllib.error.URLError as e:
        logger.error("Telegram photo send failed: %s", e)
        return False


# ── Chart Generation ────────────────────────────────────────────

def generate_chart(ticker: str) -> Optional[bytes]:
    """Generate a professional candlestick chart with indicators.

    Draws: Candlesticks, EMA-9/21, Bollinger Bands, RSI subplot,
    MACD subplot, and volume bars.

    Args:
        ticker: Stock ticker symbol (e.g., 'RELIANCE.NS').

    Returns:
        PNG image as bytes, or None if data unavailable.
    """
    from trading_assistant.data.database import DatabaseManager
    from trading_assistant.features.technical_indicators import compute_all_indicators

    db = DatabaseManager()
    try:
        df = db.get_price_data(ticker, interval="15m", limit=200)
    finally:
        db.close()

    if df.empty or len(df) < 30:
        return None

    enriched = compute_all_indicators(df)
    if enriched.empty:
        return None

    # Use the last 80 candles for a clean chart
    data = enriched.tail(80).copy()

    # ── Create figure with 4 subplots ──
    fig, (ax_price, ax_vol, ax_rsi, ax_macd) = plt.subplots(
        4, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [4, 1, 1.5, 1.5]},
        sharex=True,
    )
    fig.patch.set_facecolor("#0e1117")
    for ax in (ax_price, ax_vol, ax_rsi, ax_macd):
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")

    x = np.arange(len(data))

    # ── Candlesticks ──
    colors = ["#26a69a" if c >= o else "#ef5350"
              for o, c in zip(data["open"], data["close"])]
    ax_price.bar(x, data["close"] - data["open"], bottom=data["open"],
                 color=colors, width=0.6, edgecolor=colors, linewidth=0.5)
    ax_price.vlines(x, data["low"], data["high"], color=colors, linewidth=0.5)

    # ── Bollinger Bands ──
    if "bb_upper" in data.columns:
        ax_price.fill_between(x, data["bb_lower"], data["bb_upper"],
                              alpha=0.08, color="#42a5f5")
        ax_price.plot(x, data["bb_mid"], color="#42a5f5", linewidth=0.7,
                      alpha=0.5, linestyle="--")

    # ── EMAs ──
    if "ema_9" in data.columns:
        ax_price.plot(x, data["ema_9"], color="#ffeb3b", linewidth=1,
                      label="EMA 9", alpha=0.8)
    if "ema_21" in data.columns:
        ax_price.plot(x, data["ema_21"], color="#ff9800", linewidth=1,
                      label="EMA 21", alpha=0.8)

    ax_price.set_title(f"{ticker}", fontsize=14, fontweight="bold",
                       color="white", pad=10)
    ax_price.legend(loc="upper left", fontsize=7, facecolor="#1a1a2e",
                    edgecolor="#333", labelcolor="white")
    ax_price.set_ylabel("Price (₹)", color="#aaaaaa", fontsize=9)

    # ── Volume ──
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for o, c in zip(data["open"], data["close"])]
    ax_vol.bar(x, data["volume"], color=vol_colors, alpha=0.6, width=0.6)
    ax_vol.set_ylabel("Vol", color="#aaaaaa", fontsize=8)

    # ── RSI ──
    if "rsi_14" in data.columns:
        ax_rsi.plot(x, data["rsi_14"], color="#ab47bc", linewidth=1.2)
        ax_rsi.axhline(70, color="#ef5350", linestyle="--", linewidth=0.7, alpha=0.6)
        ax_rsi.axhline(30, color="#26a69a", linestyle="--", linewidth=0.7, alpha=0.6)
        ax_rsi.fill_between(x, 30, 70, alpha=0.05, color="#ab47bc")
        ax_rsi.set_ylabel("RSI", color="#aaaaaa", fontsize=8)
        ax_rsi.set_ylim(10, 90)

    # ── MACD ──
    if "macd_line" in data.columns:
        ax_macd.plot(x, data["macd_line"], color="#42a5f5", linewidth=1, label="MACD")
        ax_macd.plot(x, data["macd_signal"], color="#ff7043", linewidth=1,
                     label="Signal")
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350"
                       for v in data["macd_histogram"]]
        ax_macd.bar(x, data["macd_histogram"], color=hist_colors, alpha=0.5,
                    width=0.6)
        ax_macd.set_ylabel("MACD", color="#aaaaaa", fontsize=8)
        ax_macd.legend(loc="upper left", fontsize=6, facecolor="#1a1a2e",
                       edgecolor="#333", labelcolor="white")

    # ── Format x-axis with timestamps ──
    tick_positions = np.linspace(0, len(data) - 1, min(8, len(data)), dtype=int)
    ax_macd.set_xticks(tick_positions)
    labels = [data.index[i].strftime("%d %b\n%H:%M") if hasattr(data.index[i], "strftime")
              else str(data.index[i]) for i in tick_positions]
    ax_macd.set_xticklabels(labels, fontsize=7, color="#aaaaaa")

    plt.tight_layout()

    # ── Export to PNG bytes ──
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#0e1117", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Telegram Bot Polling (Commands) ─────────────────────────────

class TelegramBotPoller:
    """Simple long-polling Telegram bot for handling /commands.

    Runs in a background thread and responds to:
      /chart TICKER  — sends a candlestick chart image.
      /status        — shows current market regime and latest signals.
      /help          — lists all commands.
    """

    def __init__(self):
        self._offset = 0
        self._running = False
        self._thread = None

    def start(self):
        """Start the bot polling in a background daemon thread."""
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Telegram bot polling started")

    def stop(self):
        """Stop the polling loop."""
        self._running = False
        logger.info("Telegram bot polling stopped")

    def _poll_loop(self):
        """Main polling loop — fetches updates every 2 seconds."""
        import time
        while self._running:
            try:
                self._process_updates()
            except Exception as exc:
                logger.error("Telegram poll error: %s", exc)
            time.sleep(2)

    def _process_updates(self):
        """Fetch and process new messages from Telegram."""
        url = (f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
               f"?offset={self._offset}&timeout=1")
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
        except Exception:
            return

        if not data.get("ok") or not data.get("result"):
            return

        for update in data["result"]:
            self._offset = update["update_id"] + 1
            message = update.get("message", {})
            text = message.get("text", "").strip()
            chat_id = str(message.get("chat", {}).get("id", ""))

            # Only respond to the configured chat
            if chat_id != TELEGRAM_CHAT_ID:
                continue

            if text.startswith("/chart"):
                self._handle_chart(text)
            elif text.startswith("/status"):
                self._handle_status()
            elif text.startswith("/help") or text.startswith("/start"):
                self._handle_help()

    def _handle_chart(self, text: str):
        """Handle /chart TICKER command."""
        parts = text.split()
        if len(parts) < 2:
            send_telegram_message(
                "📊 *Usage:* `/chart RELIANCE.NS`\n\n"
                "Available tickers:\n" +
                ", ".join(f"`{t}`" for t in STOCKS.TICKERS[:10]) +
                "\n... and more"
            )
            return

        ticker = parts[1].upper()
        # Add .NS suffix if missing
        if not ticker.endswith(".NS"):
            ticker += ".NS"

        if ticker not in STOCKS.TICKERS:
            send_telegram_message(
                f"❌ *{ticker}* is not in our Nifty 50 watchlist.\n"
                f"Try one of: {', '.join(STOCKS.TICKERS[:5])}"
            )
            return

        send_telegram_message(f"📈 Generating chart for *{ticker}*... please wait.")

        chart_bytes = generate_chart(ticker)
        if chart_bytes:
            send_telegram_photo(
                chart_bytes,
                caption=f"📈 *{ticker}* — 15min Chart\n"
                        f"🕐 {datetime.now(IST).strftime('%d %b %Y, %I:%M %p IST')}",
            )
        else:
            send_telegram_message(
                f"⚠️ Not enough data for *{ticker}* yet. "
                f"Run the main pipeline first to fetch historical data."
            )

    def _handle_status(self):
        """Handle /status command — show market regime + recent signals."""
        from trading_assistant.features.market_regime import detect_regime, MarketRegime

        regime = detect_regime()
        regime_emoji = {
            MarketRegime.BULLISH: "🟢 BULLISH — All clear, BUY signals active",
            MarketRegime.CAUTIOUS: "🟡 CAUTIOUS — VIX elevated, reduced positions",
            MarketRegime.BEARISH: "🔴 BEARISH — Nifty downtrend, BUY signals blocked",
            MarketRegime.PANIC: "🚨 PANIC — Extreme fear, all BUY signals blocked",
        }

        msg = (f"📊 *MARKET STATUS*\n\n"
               f"*Regime:* {regime_emoji.get(regime, regime.value)}\n"
               f"*Time:* {datetime.now(IST).strftime('%d %b %Y, %I:%M %p IST')}\n"
               f"*Tracking:* {len(STOCKS.TICKERS)} Nifty 50 stocks\n\n"
               f"Use `/chart RELIANCE.NS` to view any stock chart.")
        send_telegram_message(msg)

    def _handle_help(self):
        """Handle /help command."""
        msg = ("🤖 *Mr. Stonk Bot — Commands*\n\n"
               "📈 `/chart RELIANCE.NS` — View candlestick chart\n"
               "📊 `/status` — Check market regime\n"
               "❓ `/help` — Show this message\n\n"
               "💡 *Tip:* You can use short names like:\n"
               "`/chart RELIANCE` (I'll add .NS automatically)")
        send_telegram_message(msg)


# ── Global bot instance ─────────────────────────────────────────
_bot = TelegramBotPoller()


def start_telegram_bot():
    """Start the Telegram bot polling in the background."""
    _bot.start()


def stop_telegram_bot():
    """Stop the Telegram bot polling."""
    _bot.stop()
