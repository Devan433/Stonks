"""
Main pipeline entry point for the Trading Assistant.

On startup:
  1. Fetch 4 years of daily + 60 days of 15-min data for all stocks.
  2. Load (or train) the ML model.

During market hours (via APScheduler):
  • Every 5 min  → refresh prices, compute indicators, generate signals.
  • Every 15 min → fetch news, run sentiment analysis.
  • Weekly (Sunday) → retrain model on fresh data.

All activity is logged to logs/trading_assistant.log with rotation.
"""

import logging
import sys
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

import joblib
import pandas as pd
import pytz

from trading_assistant.config import DATA, MARKET, MODEL, PATHS, STOCKS

logger = logging.getLogger("trading_assistant")
IST = pytz.timezone(MARKET.TIMEZONE)


# ── Logging Setup ───────────────────────────────────────────────

def setup_logging() -> None:
    """Configure rotating file + console logging."""
    PATHS.LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (10 MB, 5 backups)
    fh = RotatingFileHandler(
        PATHS.LOG_FILE, maxBytes=10_000_000, backupCount=5, encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)

    logger.info("Logging initialised → %s", PATHS.LOG_FILE)


# ── Market Hours Helper ─────────────────────────────────────────

def _is_market_hours() -> bool:
    """Check if the current time is within Indian market trading hours.

    Returns True if:
      - It is a weekday (Mon–Fri)
      - It is NOT an NSE holiday
      - The time is between 9:30 AM and 3:30 PM IST

    The 9:15–9:30 window is intentionally excluded to avoid
    volatile opening-minute signals.
    """
    now = datetime.now(IST)
    weekday = now.weekday()  # 0=Mon, 6=Sun

    if weekday > 4:  # Saturday or Sunday
        return False

    # Check NSE holidays
    today_str = now.strftime("%Y-%m-%d")
    if today_str in MARKET.NSE_HOLIDAYS:
        return False

    market_start = now.replace(hour=MARKET.NO_TRADE_END[0],
                               minute=MARKET.NO_TRADE_END[1], second=0)
    market_end = now.replace(hour=MARKET.MARKET_CLOSE[0],
                             minute=MARKET.MARKET_CLOSE[1], second=0)

    return market_start <= now <= market_end


def _next_market_open() -> str:
    """Return a human-readable string for when the market opens next."""
    now = datetime.now(IST)
    weekday = now.weekday()

    if weekday < 4:  # Mon–Thu
        next_day = now.replace(hour=9, minute=15, second=0) \
            if now.hour < 9 or (now.hour == 9 and now.minute < 15) \
            else now.replace(hour=9, minute=15, second=0) + timedelta(days=1)
    elif weekday == 4:  # Friday
        if now.hour < 15 or (now.hour == 15 and now.minute < 30):
            next_day = now.replace(hour=9, minute=15, second=0) \
                if now.hour < 9 or (now.hour == 9 and now.minute < 15) \
                else now  # market is open right now
        else:
            next_day = now + timedelta(days=3)  # skip to Monday
            next_day = next_day.replace(hour=9, minute=15, second=0)
    elif weekday == 5:  # Saturday
        next_day = now + timedelta(days=2)
        next_day = next_day.replace(hour=9, minute=15, second=0)
    else:  # Sunday
        next_day = now + timedelta(days=1)
        next_day = next_day.replace(hour=9, minute=15, second=0)

    return next_day.strftime("%A, %d %b %Y at %I:%M %p IST")


# ── Pipeline Steps ──────────────────────────────────────────────

def initial_data_load() -> None:
    """Fetch all historical data on first run."""
    from trading_assistant.data.fetch_price_data import fetch_all_historical

    logger.info("=== Initial Data Load ===")
    logger.info("Fetching %d years daily + %d days intraday for %d stocks",
                DATA.HISTORICAL_YEARS, DATA.INTRADAY_DAYS, len(STOCKS.TICKERS))
    fetch_all_historical()
    logger.info("Initial data load complete")


def refresh_prices() -> None:
    """Scheduled job: refresh 15-min price data and generate signals."""
    if not _is_market_hours():
        logger.debug("Skipping price refresh — market is closed")
        return

    from trading_assistant.data.database import DatabaseManager
    from trading_assistant.data.fetch_price_data import refresh_live_data
    from trading_assistant.features.technical_indicators import compute_all_indicators
    from trading_assistant.features.market_regime import (
        detect_regime, should_allow_buy, get_position_scale, MarketRegime,
    )

    logger.info("--- Price Refresh ---")

    # ── Market Regime Check ──
    regime = detect_regime()
    allow_buy = should_allow_buy(regime)
    pos_scale = get_position_scale(regime)
    logger.info("Market Regime: %s | BUY allowed: %s | Position scale: %.0f%%",
                regime.value, allow_buy, pos_scale * 100)

    results = refresh_live_data()

    if not results:
        return

    # Compute indicators and generate signals for each stock
    db = DatabaseManager()
    try:
        model = _load_model()
        if model is None:
            logger.warning("No trained model — skipping signal generation")
            return

        for ticker, df in results.items():
            try:
                full_df = db.get_price_data(ticker, interval="15m", limit=200)
                if len(full_df) < 30:
                    continue

                enriched = compute_all_indicators(full_df)
                if enriched.empty:
                    continue

                # ── Merge sentiment score into features ──
                sent_df = db.get_sentiment(ticker)
                if not sent_df.empty:
                    sentiment_val = float(sent_df["avg_sentiment"].iloc[-1])
                else:
                    sentiment_val = 0.0
                enriched["sentiment_score"] = sentiment_val

                signal = _predict_signal(model, enriched, ticker)
                if signal:
                    # ── Regime Filter: block BUY in bearish/panic ──
                    if signal["signal"] == "BUY" and not allow_buy:
                        logger.info(
                            "🛑 BUY signal for %s BLOCKED by %s regime",
                            ticker, regime.value,
                        )
                        from trading_assistant.notifications import send_telegram_message
                        send_telegram_message(
                            f"🛑 *BUY BLOCKED: {ticker}*\n\n"
                            f"The AI found a BUY setup, but the *Market Regime "
                            f"is {regime.value}*.\n"
                            f"India VIX is elevated or Nifty is in a downtrend. "
                            f"Protecting your capital. ✋"
                        )
                        continue

                    db.store_trade_signal(signal)
                    logger.info("Signal for %s: %s (%.1f%%)",
                                ticker, signal["signal"],
                                (signal.get("confidence", 0) or 0) * 100)
                                
                    if signal["signal"] in ("BUY", "SELL"):
                        from trading_assistant.notifications import send_telegram_message
                        
                        # Extract latest features for the notification
                        latest = enriched.iloc[-1]
                        entry = latest["close"]
                        atr = latest["atr_14"]
                        rsi = latest["rsi_14"]
                        macd = latest["macd_line"]
                        vol_spike = "YES" if latest.get("volume_spike", 0) else "NO"

                        # SMC features
                        fvg_bull = "YES" if latest.get("fvg_bullish", 0) else "—"
                        fvg_bear = "YES" if latest.get("fvg_bearish", 0) else "—"
                        ob_bull = "YES" if latest.get("ob_bullish", 0) else "—"
                        ob_bear = "YES" if latest.get("ob_bearish", 0) else "—"
                        sweep_bull = "YES" if latest.get("sweep_bullish", 0) else "—"
                        sweep_bear = "YES" if latest.get("sweep_bearish", 0) else "—"
                        
                        # Sentiment value (already fetched above)
                        sentiment = f"{sentiment_val:.2f}"
                        
                        # Calculate targets
                        if signal["signal"] == "BUY":
                            sl = entry - (1.5 * atr)
                            tp = entry + (3.0 * atr)
                            sl_pct = (sl - entry) / entry * 100
                            tp_pct = (tp - entry) / entry * 100
                        else:
                            sl = entry + (1.5 * atr)
                            tp = entry - (3.0 * atr)
                            sl_pct = (sl - entry) / entry * 100
                            tp_pct = (tp - entry) / entry * 100

                        # Regime status
                        regime_status = {
                            MarketRegime.BULLISH: "BULLISH",
                            MarketRegime.CAUTIOUS: "CAUTIOUS",
                            MarketRegime.BEARISH: "BEARISH",
                            MarketRegime.PANIC: "PANIC",
                        }

                        # Position sizing guidance
                        pos_pct = int(pos_scale * 100)
                        if pos_scale >= 1.0:
                            pos_guidance = f"Full position ({pos_pct}%)"
                        elif pos_scale > 0:
                            pos_guidance = f"Reduced position ({pos_pct}%) — {regime.value} regime"
                        else:
                            pos_guidance = f"No position — {regime.value} regime"

                        # Fetch recent news for context
                        news_df = db.get_recent_news(hours=24, symbol=ticker)
                        news_text = ""
                        if not news_df.empty:
                            news_text = "\n*RECENT NEWS*\n"
                            for _, row in news_df.head(3).iterrows():
                                news_text += f"• {row['headline']}\n"
                        else:
                            news_text = "\n*RECENT NEWS*\n• No major news in last 24h.\n"

                        msg = (f"*{signal['signal']} SIGNAL: {ticker}*\n\n"
                               f"*TRADE SETUP*\n"
                               f"• Entry Price: ₹{entry:,.2f}\n"
                               f"• Take Profit: ₹{tp:,.2f} ({tp_pct:+.1f}%)\n"
                               f"• Stop Loss: ₹{sl:,.2f} ({sl_pct:+.1f}%)\n"
                               f"• Position Size: {pos_guidance}\n\n"
                               f"*AI PREDICTION*\n"
                               f"• Confidence: {signal.get('confidence', 0):.1%}\n"
                               f"• Market Regime: {regime_status.get(regime, regime.value)}\n\n"
                               f"*TECHNICAL REASONS*\n"
                               f"• MACD: {macd:,.2f}\n"
                               f"• RSI (14): {rsi:.1f}\n"
                               f"• Volume Spike: {vol_spike}\n"
                               f"• News Sentiment: {sentiment}\n\n"
                               f"*SMART MONEY*\n"
                               f"• Bullish FVG: {fvg_bull} | Bearish FVG: {fvg_bear}\n"
                               f"• Bullish OB: {ob_bull} | Bearish OB: {ob_bear}\n"
                               f"• Bull Sweep: {sweep_bull} | Bear Sweep: {sweep_bear}\n"
                               f"{news_text}")
                        send_telegram_message(msg)
            except Exception as exc:
                logger.error("Signal gen failed for %s: %s", ticker, exc)
    finally:
        db.close()


def refresh_news() -> None:
    """Scheduled job: fetch news and run sentiment analysis."""
    if not _is_market_hours():
        logger.debug("Skipping news refresh — market is closed")
        return

    from trading_assistant.data.fetch_news import fetch_all_news
    from trading_assistant.features.sentiment_analysis import (
        get_analyzer, score_articles,
    )
    from trading_assistant.data.database import DatabaseManager

    logger.info("--- News Refresh ---")
    articles = fetch_all_news(store=True)

    if articles:
        analyzer = get_analyzer("finbert")
        scores = score_articles(articles, analyzer)
        if scores:
            with DatabaseManager() as db:
                db.store_sentiment_scores(scores)
            logger.info("Scored %d sentiment records", len(scores))


def retrain_model() -> None:
    """Scheduled job: retrain the ML model on latest data."""
    import pandas as pd
    from trading_assistant.data.database import DatabaseManager
    from trading_assistant.features.technical_indicators import compute_all_indicators
    from trading_assistant.models.train_model import run_training_pipeline

    logger.info("=== Weekly Model Retrain ===")
    db = DatabaseManager()
    try:
        all_dfs = []
        for ticker in STOCKS.TICKERS:
            df = db.get_price_data(ticker, interval="15m")
            if not df.empty:
                df["symbol"] = ticker
                df["interval"] = "15m"
                enriched = compute_all_indicators(df)
                if not enriched.empty:
                    all_dfs.append(enriched)

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=False)
            results = run_training_pipeline(combined, save=True)
            logger.info("Retrain complete — best model: %s (%.4f acc)",
                        results.get("best_model"),
                        results.get(results.get("best_model", ""), {})
                              .get("avg_accuracy", 0))
        else:
            logger.warning("No data available for retraining")
    finally:
        db.close()


def cleanup_database() -> None:
    """Scheduled job: remove old data to prevent database bloat.

    Deletes:
      - Price data older than 6 months
      - News articles older than 30 days
      - Sentiment scores older than 30 days
    Trade signals are kept forever for auditing.
    """
    from trading_assistant.data.database import DatabaseManager

    logger.info("=== Database Cleanup ===")
    db = DatabaseManager()
    try:
        cur = db.conn.cursor()

        # Delete price data older than 6 months
        cur.execute(
            "DELETE FROM price_data WHERE timestamp < datetime('now', '-6 months')"
        )
        price_deleted = cur.rowcount

        # Delete news older than 30 days
        cur.execute(
            "DELETE FROM news_articles WHERE fetched_at < datetime('now', '-30 days')"
        )
        news_deleted = cur.rowcount

        # Delete sentiment scores older than 30 days
        cur.execute(
            "DELETE FROM sentiment_scores WHERE created_at < datetime('now', '-30 days')"
        )
        sent_deleted = cur.rowcount

        db.conn.commit()

        # Reclaim disk space
        db.conn.execute("VACUUM")

        logger.info("Cleanup complete — removed %d price rows, %d news, %d sentiment records",
                     price_deleted, news_deleted, sent_deleted)
    except Exception as exc:
        logger.error("Database cleanup failed: %s", exc)
    finally:
        db.close()


def check_signal_accuracy() -> None:
    """Scheduled job: verify past signals against actual price movements.

    Checks BUY/SELL signals from 1 hour ago and compares the predicted
    direction with what actually happened. Logs the result and updates
    the trade_signals table with accuracy data.
    """
    from trading_assistant.data.database import DatabaseManager

    logger.info("--- Signal Accuracy Check ---")
    db = DatabaseManager()
    try:
        # Get signals from ~1 hour ago
        signals = pd.read_sql_query(
            "SELECT id, symbol, signal, confidence, timestamp "
            "FROM trade_signals "
            "WHERE signal IN ('BUY', 'SELL') "
            "AND timestamp <= datetime('now', '-1 hour') "
            "AND timestamp > datetime('now', '-2 hours')",
            db.conn,
        )

        if signals.empty:
            logger.debug("No signals to verify from the last hour")
            return

        correct = 0
        total = 0

        for _, row in signals.iterrows():
            # Get price at signal time and 1 hour later
            price_data = db.get_price_data(
                row["symbol"], interval="15m", limit=10,
            )
            if len(price_data) < 2:
                continue

            signal_price = price_data.iloc[-5]["close"] if len(price_data) >= 5 else price_data.iloc[0]["close"]
            current_price = price_data.iloc[-1]["close"]
            pct_change = (current_price - signal_price) / signal_price * 100

            was_correct = (
                (row["signal"] == "BUY" and pct_change > 0) or
                (row["signal"] == "SELL" and pct_change < 0)
            )

            if was_correct:
                correct += 1
            total += 1

            logger.info("Signal verify: %s %s → %.2f%% move → %s",
                        row["signal"], row["symbol"], pct_change,
                        "CORRECT ✅" if was_correct else "WRONG ❌")

        if total > 0:
            accuracy = correct / total * 100
            logger.info("Signal accuracy (last hour): %d/%d (%.1f%%)",
                        correct, total, accuracy)
    except Exception as exc:
        logger.error("Signal accuracy check failed: %s", exc)
    finally:
        db.close()


# ── Model Loading & Prediction ──────────────────────────────────

def _load_model():
    """Load the best saved model from disk."""
    for name in ("xgboost_best.pkl", "random_forest_best.pkl"):
        path = PATHS.MODEL_DIR / name
        if path.exists():
            logger.info("Loading model from %s", path)
            return joblib.load(path)
    return None


def _predict_signal(model, df, ticker: str) -> dict:
    """Generate a prediction signal from the latest indicator row."""
    from trading_assistant.models.train_model import add_time_features

    df = add_time_features(df)

    drop_cols = ["open", "high", "low", "close", "volume",
                 "symbol", "interval", "target"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    latest = df[feature_cols].iloc[[-1]].fillna(0)

    try:
        pred = model.predict(latest)[0]
        proba = model.predict_proba(latest)[0]
        le = model._label_encoder
        label = le.inverse_transform([pred])[0]
        confidence = float(max(proba))

        signal_map = {"UP": "BUY", "DOWN": "SELL", "SIDEWAYS": "HOLD"}
        raw_signal = signal_map.get(label, "HOLD")

        # Downgrade low-confidence BUY/SELL to HOLD
        if raw_signal in ("BUY", "SELL") and confidence < MODEL.MIN_SIGNAL_CONFIDENCE:
            logger.info("Signal %s for %s downgraded to HOLD (confidence %.1f%% < %.0f%% threshold)",
                        raw_signal, ticker, confidence * 100, MODEL.MIN_SIGNAL_CONFIDENCE * 100)
            raw_signal = "HOLD"

        return {
            "timestamp": datetime.now(IST).isoformat(),
            "symbol": ticker,
            "signal": raw_signal,
            "confidence": confidence,
            "model_name": type(model).__name__,
            "features_snapshot": None,
        }
    except Exception as exc:
        logger.error("Prediction failed for %s: %s", ticker, exc)
        return {}


# ── Scheduler ───────────────────────────────────────────────────

def start_scheduler() -> None:
    """Start APScheduler with all recurring jobs."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler = BlockingScheduler(timezone=IST)

    # Price refresh every 5 min, Mon-Fri, 9:15–15:30
    scheduler.add_job(
        refresh_prices,
        IntervalTrigger(minutes=DATA.PRICE_REFRESH_MINS),
        id="price_refresh",
        name="Price Data Refresh",
        max_instances=1,
    )

    # News refresh every 15 min, Mon-Fri
    scheduler.add_job(
        refresh_news,
        IntervalTrigger(minutes=DATA.NEWS_REFRESH_MINS),
        id="news_refresh",
        name="News Refresh",
        max_instances=1,
    )

    # Weekly retrain on Sunday at midnight IST
    scheduler.add_job(
        retrain_model,
        CronTrigger(day_of_week="sun", hour=0, minute=0),
        id="weekly_retrain",
        name="Weekly Model Retrain",
        max_instances=1,
    )

    # Signal accuracy check every 1 hour during market hours
    scheduler.add_job(
        check_signal_accuracy,
        IntervalTrigger(hours=1),
        id="accuracy_check",
        name="Signal Accuracy Check",
        max_instances=1,
    )

    # Database cleanup on 1st of every month at midnight
    scheduler.add_job(
        cleanup_database,
        CronTrigger(day=1, hour=0, minute=0),
        id="db_cleanup",
        name="Monthly Database Cleanup",
        max_instances=1,
    )

    logger.info("Scheduler started with 5 jobs. Press Ctrl+C to exit.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shutting down…")
        _print_session_summary()
        logger.info("Scheduler shut down gracefully")


def _print_session_summary() -> None:
    """Print a session summary on shutdown."""
    from trading_assistant.data.database import DatabaseManager
    from trading_assistant.notifications import send_telegram_message

    try:
        db = DatabaseManager()
        today = datetime.now(IST).strftime("%Y-%m-%d")

        # Count today's signals
        signals = pd.read_sql_query(
            "SELECT signal, COUNT(*) as cnt FROM trade_signals "
            "WHERE timestamp LIKE ? GROUP BY signal",
            db.conn, params=[f"{today}%"],
        )
        db.close()

        buy_count = 0
        sell_count = 0
        hold_count = 0
        for _, row in signals.iterrows():
            if row["signal"] == "BUY":
                buy_count = row["cnt"]
            elif row["signal"] == "SELL":
                sell_count = row["cnt"]
            elif row["signal"] == "HOLD":
                hold_count = row["cnt"]

        total = buy_count + sell_count + hold_count

        summary = (
            f"📴 *Trading Assistant Shutting Down*\n\n"
            f"📅 *Session Date:* {today}\n"
            f"📊 *Signals Generated:* {total}\n"
            f"  • BUY: {buy_count}\n"
            f"  • SELL: {sell_count}\n"
            f"  • HOLD: {hold_count}\n\n"
            f"See you next session! 👋"
        )

        logger.info("Session summary — BUY: %d, SELL: %d, HOLD: %d (total: %d)",
                     buy_count, sell_count, hold_count, total)
        send_telegram_message(summary)

    except Exception as exc:
        logger.error("Failed to generate session summary: %s", exc)


# ── Entry Point ─────────────────────────────────────────────────

def main() -> None:
    """Run the full trading assistant pipeline."""
    import pandas as pd  # noqa: ensure pandas is importable
    from trading_assistant.notifications import start_telegram_bot, send_telegram_message
    from trading_assistant.features.market_regime import detect_regime

    setup_logging()
    logger.info("=" * 60)
    logger.info("  Indian Stock Market Trading Assistant v2.0")
    logger.info("  Tracking %d Nifty 50 stocks", len(STOCKS.TICKERS))
    logger.info("=" * 60)

    # Step 0: Start Telegram bot (background thread for /chart commands)
    start_telegram_bot()
    logger.info("Telegram bot started — listening for /chart and /status commands")

    # Step 1: Initial data load
    initial_data_load()

    # Step 2: Initial news fetch + sentiment
    refresh_news()

    # Step 3: Load or train model
    model = _load_model()
    if model is None:
        logger.info("No saved model found — training from scratch")
        retrain_model()

    # Step 4: Check initial market regime
    regime = detect_regime()
    logger.info("Initial Market Regime: %s", regime.value)

    # Step 5: Notify user that the bot is live
    send_telegram_message(
        "🟢 *Trading Assistant v2.0 is LIVE!*\n\n"
        f"📊 Tracking *{len(STOCKS.TICKERS)} Nifty 50* stocks\n"
        f"🛡️ Market Regime: *{regime.value}*\n"
        f"🤖 AI Model loaded and ready\n\n"
        "Commands:\n"
        "📈 `/chart RELIANCE.NS` — View chart\n"
        "📊 `/status` — Market regime\n"
        "❓ `/help` — All commands"
    )

    # Step 6: Start scheduler for live updates
    if _is_market_hours():
        logger.info("Market is OPEN — starting live signal generation")
    else:
        next_open = _next_market_open()
        logger.info("Market is CLOSED — bot will auto-activate on %s", next_open)
        send_telegram_message(
            f"💤 *Market is currently closed.*\n\n"
            f"The bot is running and will automatically start "
            f"scanning for signals when the market opens next:\n"
            f"📅 *{next_open}*\n\n"
            f"Telegram commands (`/chart`, `/status`) still work!"
        )

    logger.info("Starting scheduled jobs…")
    start_scheduler()


if __name__ == "__main__":
    main()
