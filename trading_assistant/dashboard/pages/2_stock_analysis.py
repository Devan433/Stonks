import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trading_assistant.data.database import DatabaseManager
from trading_assistant.features.technical_indicators import compute_all_indicators
from trading_assistant.config import STOCKS

st.set_page_config(page_title="Stock Analysis", layout="wide")
st.title("Stock Analysis")

# Select Stock
ticker = st.selectbox("Select a Nifty 50 Stock", STOCKS.TICKERS)

db = DatabaseManager()
try:
    with st.spinner("Fetching data..."):
        df = db.get_price_data(ticker, interval="15m", limit=300)
        sent_df = db.get_sentiment(ticker)
        
    if df.empty or len(df) < 50:
        st.warning(f"Not enough data for {ticker}. Run the pipeline first.")
    else:
        enriched = compute_all_indicators(df)
        latest = enriched.iloc[-1]
        
        # 1. Top Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Close Price", f"₹{latest['close']:,.2f}", f"{latest['close'] - enriched.iloc[-2]['close']:.2f}")
        col2.metric("RSI (14)", f"{latest['rsi_14']:.1f}")
        col3.metric("MACD Line", f"{latest['macd_line']:.2f}")
        
        sentiment_score = float(sent_df["avg_sentiment"].iloc[-1]) if not sent_df.empty else 0.0
        sent_label = "Positive" if sentiment_score > 0.2 else ("Negative" if sentiment_score < -0.2 else "Neutral")
        col4.metric("News Sentiment", sent_label, f"{sentiment_score:.2f}")

        # 2. SMC Footprints
        st.markdown("### Smart Money Concepts (Latest Candle)")
        smc_col1, smc_col2, smc_col3 = st.columns(3)
        smc_col1.write(f"**Bullish FVG:** {'Detected' if latest.get('fvg_bullish', 0) else '-'}")
        smc_col1.write(f"**Bearish FVG:** {'Detected' if latest.get('fvg_bearish', 0) else '-'}")
        smc_col2.write(f"**Bullish OB:** {'Detected' if latest.get('ob_bullish', 0) else '-'}")
        smc_col2.write(f"**Bearish OB:** {'Detected' if latest.get('ob_bearish', 0) else '-'}")
        smc_col3.write(f"**Bullish Sweep:** {'Detected' if latest.get('sweep_bullish', 0) else '-'}")
        smc_col3.write(f"**Bearish Sweep:** {'Detected' if latest.get('sweep_bearish', 0) else '-'}")
        
        # 3. Interactive Chart
        st.markdown("### Interactive Chart (Last 100 Candles)")
        plot_data = enriched.tail(100)
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=('Price & EMAs', 'Volume', 'MACD & RSI'),
                            row_width=[0.2, 0.2, 0.6])

        # Candlestick
        fig.add_trace(go.Candlestick(x=plot_data.index,
                                     open=plot_data['open'],
                                     high=plot_data['high'],
                                     low=plot_data['low'],
                                     close=plot_data['close'],
                                     name='Price'), row=1, col=1)
        
        # EMAs
        if 'ema_9' in plot_data.columns:
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['ema_9'], name='EMA 9', line=dict(color='yellow', width=1)), row=1, col=1)
        if 'ema_21' in plot_data.columns:
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['ema_21'], name='EMA 21', line=dict(color='orange', width=1)), row=1, col=1)

        # Volume
        colors = ['green' if row['close'] >= row['open'] else 'red' for idx, row in plot_data.iterrows()]
        fig.add_trace(go.Bar(x=plot_data.index, y=plot_data['volume'], marker_color=colors, name='Volume'), row=2, col=1)

        # MACD
        if 'macd_line' in plot_data.columns:
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['macd_line'], name='MACD Line', line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['macd_signal'], name='MACD Signal', line=dict(color='orange')), row=3, col=1)
            fig.add_trace(go.Bar(x=plot_data.index, y=plot_data['macd_histogram'], name='MACD Hist'), row=3, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

finally:
    db.close()
