import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path
import joblib

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trading_assistant.data.database import DatabaseManager
from trading_assistant.features.technical_indicators import compute_all_indicators
from trading_assistant.models.train_model import add_time_features
from trading_assistant.backtesting.backtest import run_backtest
from trading_assistant.config import STOCKS, PATHS

st.set_page_config(page_title="Strategy Backtesting", layout="wide")
st.title("Strategy Backtesting")
st.write("Run the active ML model against historical data to evaluate performance metrics.")

col1, col2 = st.columns([1, 3])

with col1:
    ticker = st.selectbox("Select Stock", STOCKS.TICKERS)
    run_btn = st.button("Run Backtest", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Methodology")
    st.write("The backtesting engine fetches 15-minute historical data, computes indicators, and simulates trading with a ₹1,00,000 initial balance using the latest production model.")

if run_btn:
    with col2:
        with st.spinner(f"Running backtest for {ticker}..."):
            # Load model
            model = None
            for name in ("xgboost_best.pkl", "random_forest_best.pkl"):
                path = PATHS.MODEL_DIR / name
                if path.exists():
                    model = joblib.load(path)
                    break
            
            if model is None:
                st.error("No trained model found. Please wait for the retraining pipeline to finish.")
            else:
                db = DatabaseManager()
                try:
                    df = db.get_price_data(ticker, interval="15m")
                finally:
                    db.close()

                if df.empty or len(df) < 50:
                    st.error(f"Not enough historical data for {ticker}.")
                else:
                    enriched = compute_all_indicators(df)
                    enriched = add_time_features(enriched)

                    # Generate predictions
                    drop_cols = ["open", "high", "low", "close", "volume", "symbol", "interval", "target"]
                    feature_cols = [c for c in enriched.columns if c not in drop_cols]
                    X = enriched[feature_cols].fillna(0)

                    preds = model.predict(X)
                    le = model._label_encoder
                    labels = le.inverse_transform(preds)
                    
                    signal_map = {"UP": 2, "DOWN": 0, "SIDEWAYS": 1}
                    signals = pd.Series([signal_map.get(l, 1) for l in labels], index=enriched.index)

                    # Run engine
                    results = run_backtest(enriched, signals)
                    m = results["metrics"]
                    eq_df = results["equity_curve"]

                    st.success(f"Backtest completed successfully! ({len(enriched)} candles)")
                    
                    # Metrics Row
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Return", f"{m['total_return_pct']:+.2f}%")
                    m2.metric("Win Rate", f"{m['win_rate_pct']:.1f}%")
                    m3.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
                    m4.metric("Max Drawdown", f"{m['max_drawdown_pct']:.2f}%")
                    
                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("Final Equity", f"₹{m['final_equity']:,.0f}")
                    m6.metric("Total Trades", m['n_trades'])
                    m7.metric("Avg Trade %", f"{m['avg_trade_pct']:+.2f}%")
                    m8.metric("Profit Factor", f"{m['profit_factor']:.2f}")

                    # Equity Curve Chart
                    st.markdown("### Equity Curve")
                    fig = px.line(eq_df, x=eq_df.index, y="Equity", 
                                  title="Portfolio Equity Over Time",
                                  template="plotly_dark")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
