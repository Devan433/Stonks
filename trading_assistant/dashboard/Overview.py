import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import json
from pathlib import Path

# Add project root to sys.path so we can import trading_assistant
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trading_assistant.data.database import DatabaseManager
from trading_assistant.features.market_regime import detect_regime, MarketRegime
import pytz
from trading_assistant.config import MARKET

IST = pytz.timezone(MARKET.TIMEZONE)

st.set_page_config(page_title="Market Overview", layout="wide")
st.title("Market Overview")

# 1. Market Regime
regime = detect_regime()
regime_status = regime.value.upper()

st.header("Market Regime")
st.info(f"**Status:** {regime_status}")

if regime == MarketRegime.BULLISH:
    st.write("Constraint: BUY signals permitted (100% position sizing).")
elif regime == MarketRegime.CAUTIOUS:
    st.write("Constraint: BUY signals permitted (Reduced position sizing due to elevated VIX).")
else:
    st.write("Constraint: BUY signals blocked (Downtrend/High Volatility).")

st.markdown("---")

# 2. Today's Signals
today = datetime.now(IST).strftime("%Y-%m-%d")
st.header(f"System Signals ({today})")

db = DatabaseManager()
try:
    signals = pd.read_sql_query(
        "SELECT signal, COUNT(*) as cnt FROM trade_signals "
        "WHERE timestamp LIKE ? GROUP BY signal",
        db.conn, params=[f"{today}%"],
    )

    buy_cnt = sell_cnt = hold_cnt = 0
    for _, row in signals.iterrows():
        if row["signal"] == "BUY":
            buy_cnt = int(row["cnt"])
        elif row["signal"] == "SELL":
            sell_cnt = int(row["cnt"])
        elif row["signal"] == "HOLD":
            hold_cnt = int(row["cnt"])

    col1, col2, col3 = st.columns(3)
    col1.metric("BUY Signals", buy_cnt)
    col2.metric("SELL Signals", sell_cnt)
    col3.metric("HOLD Signals", hold_cnt)

    st.subheader("All Today's Signals")
    # Fetch ALL signals for today, ordered by confidence
    top_signals = pd.read_sql_query(
        "SELECT id, symbol, signal, confidence, timestamp FROM trade_signals "
        "WHERE timestamp LIKE ? "
        "ORDER BY confidence DESC",
        db.conn, params=[f"{today}%"],
    )
    
    if not top_signals.empty:
        # Keep original dataframe for lookup
        display_df = top_signals.copy()
        
        # Format for display
        display_df.rename(columns={'symbol': 'Symbol', 'signal': 'Signal', 'confidence': 'Confidence', 'timestamp': 'Timestamp'}, inplace=True)
        display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.1%}")
        
        # Color code signals (Streamlit interactive tables don't support raw CSS, so we use professional status dots)
        def add_color(sig):
            if sig == 'BUY': return '🟢 BUY'
            elif sig == 'SELL': return '🔴 SELL'
            else: return '⚪ HOLD'
            
        display_df["Signal"] = display_df["Signal"].apply(add_color)
        
        # We drop the 'id' from the display so the user doesn't see the internal DB id
        display_df_no_id = display_df.drop(columns=['id'])
        
        # Make the dataframe interactive!
        event = st.dataframe(
            display_df_no_id, 
            use_container_width=True, 
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # Handle the click event
        if len(event.selection.rows) > 0:
            selected_idx = event.selection.rows[0]
            selected_id = int(top_signals.iloc[selected_idx]['id'])
            selected_ticker = top_signals.iloc[selected_idx]['symbol']
            
            # Query the database for the detailed features snapshot for this exact signal
            signal_detail = pd.read_sql_query(
                "SELECT features_snapshot FROM trade_signals WHERE id = ?",
                db.conn, params=[selected_id]
            )
            
            if not signal_detail.empty and signal_detail.iloc[0]['features_snapshot']:
                features_json = signal_detail.iloc[0]['features_snapshot']
                try:
                    latest = json.loads(features_json)
                    
                    # Calculate targets for the UI exactly like main.py
                    entry = latest.get("close", 0)
                    atr = latest.get("atr_14", 0)
                    signal_type = top_signals.iloc[selected_idx]['signal']
                    confidence = top_signals.iloc[selected_idx]['confidence']
                    
                    if signal_type == "BUY":
                        sl = entry - (1.5 * atr)
                        tp = entry + (3.0 * atr)
                    else:
                        sl = entry + (1.5 * atr)
                        tp = entry - (3.0 * atr)
                        
                    sl_pct = ((sl - entry) / entry * 100) if entry > 0 else 0
                    tp_pct = ((tp - entry) / entry * 100) if entry > 0 else 0
                    
                    # Extract variables
                    rsi = latest.get("rsi_14", 0)
                    macd = latest.get("macd_line", 0)
                    vol_spike = "YES" if latest.get("volume_spike", 0) else "NO"
                    fvg_bull = "YES" if latest.get("fvg_bullish", 0) else "—"
                    fvg_bear = "YES" if latest.get("fvg_bearish", 0) else "—"
                    ob_bull = "YES" if latest.get("ob_bullish", 0) else "—"
                    ob_bear = "YES" if latest.get("ob_bearish", 0) else "—"
                    sweep_bull = "YES" if latest.get("sweep_bullish", 0) else "—"
                    sweep_bear = "YES" if latest.get("sweep_bearish", 0) else "—"
                    sentiment = f"{latest.get('sentiment_score', 0):.2f}"
                    
                    st.markdown("---")
                    st.subheader(f"Detailed Analysis: {selected_ticker}")
                    
                    # Render the Telegram-style card using columns
                    d_col1, d_col2 = st.columns(2)
                    
                    with d_col1:
                        st.markdown(f"**{signal_type} SIGNAL: {selected_ticker}**")
                        if signal_type != "HOLD":
                            st.markdown("**TRADE SETUP**")
                            st.markdown(f"• Entry Price: ₹{entry:,.2f}")
                            st.markdown(f"• Take Profit: ₹{tp:,.2f} ({tp_pct:+.1f}%)")
                            st.markdown(f"• Stop Loss: ₹{sl:,.2f} ({sl_pct:+.1f}%)")
                        else:
                            st.markdown("**TRADE SETUP**")
                            st.markdown(f"• Entry Price: ₹{entry:,.2f}")
                            st.markdown("• Action: No Trade / Flat")
                            
                        st.markdown("**AI PREDICTION**")
                        st.markdown(f"• Confidence: {confidence:.1%}")
                        st.markdown(f"• Market Regime: {regime_status}")
                        
                    with d_col2:
                        st.markdown("**TECHNICAL REASONS**")
                        st.markdown(f"• MACD: {macd:,.2f}")
                        st.markdown(f"• RSI (14): {rsi:.1f}")
                        st.markdown(f"• Volume Spike: {vol_spike}")
                        st.markdown(f"• News Sentiment: {sentiment}")
                        
                        st.markdown("**SMART MONEY**")
                        st.markdown(f"• Bullish FVG: {fvg_bull} | Bearish FVG: {fvg_bear}")
                        st.markdown(f"• Bullish OB: {ob_bull} | Bearish OB: {ob_bear}")
                        st.markdown(f"• Bull Sweep: {sweep_bull} | Bear Sweep: {sweep_bear}")
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.info(f"Navigate to the **Stock Analysis** page to view the live charts for {selected_ticker}.")
                
                except json.JSONDecodeError:
                    st.error("Could not parse signal features.")
            else:
                st.warning("No detailed features snapshot available for this test signal.")

    else:
        st.write("No BUY/SELL signals recorded today.")

finally:
    db.close()
