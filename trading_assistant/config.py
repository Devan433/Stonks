"""
Trading Assistant Configuration.

Centralizes all settings using frozen dataclasses: stock universe, market hours,
data fetching parameters, ML hyperparameters, trading constraints, and file paths.
All timestamps use IST (Asia/Kolkata).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ── Base directory (project root) ───────────────────────────────
BASE_DIR = Path(__file__).resolve().parent


# ── Stock Universe ──────────────────────────────────────────────
@dataclass(frozen=True)
class StockConfig:
    """Full Nifty-50 stock universe with company-name mapping for news matching."""

    TICKERS: Tuple[str, ...] = (
        # ── IT ──
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIMindtree.NS",
        # ── Banking & Finance ──
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
        # ── Energy & Oil ──
        "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "ADANIGREEN.NS",
        "COALINDIA.NS", "TATAPOWER.NS",
        # ── Auto ──
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
        "HEROMOTOCO.NS",
        # ── FMCG & Consumer ──
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS",
        # ── Metals & Mining ──
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "ADANIENT.NS",
        # ── Pharma & Healthcare ──
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "DIVISLAB.NS",
        # ── Telecom & Media ──
        "BHARTIARTL.NS",
        # ── Infra & Cement ──
        "ULTRACEMCO.NS", "GRASIM.NS", "ADANIPORTS.NS", "SHRIRAMFIN.NS",
        "SBILIFE.NS", "HDFCLIFE.NS",
        # ── Others ──
        "TITAN.NS", "ASIANPAINT.NS", "TRENT.NS", "BEL.NS",
    )

    # Maps company name variants → ticker (used by news matcher)
    COMPANY_NAMES: Dict[str, str] = field(default_factory=lambda: {
        # IT
        "TCS": "TCS.NS", "Tata Consultancy": "TCS.NS",
        "Infosys": "INFY.NS",
        "Wipro": "WIPRO.NS",
        "HCL Tech": "HCLTECH.NS", "HCL Technologies": "HCLTECH.NS",
        "Tech Mahindra": "TECHM.NS",
        "LTIMindtree": "LTIMindtree.NS", "LTI Mindtree": "LTIMindtree.NS",
        # Banking & Finance
        "HDFC Bank": "HDFCBANK.NS", "HDFC": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS", "ICICI": "ICICIBANK.NS",
        "SBI": "SBIN.NS", "State Bank": "SBIN.NS",
        "Kotak": "KOTAKBANK.NS", "Kotak Bank": "KOTAKBANK.NS",
        "Axis Bank": "AXISBANK.NS", "Axis": "AXISBANK.NS",
        "IndusInd": "INDUSINDBK.NS", "IndusInd Bank": "INDUSINDBK.NS",
        "Bajaj Finance": "BAJFINANCE.NS",
        "Bajaj Finserv": "BAJAJFINSV.NS",
        # Energy
        "Reliance": "RELIANCE.NS", "Reliance Industries": "RELIANCE.NS",
        "ONGC": "ONGC.NS", "Oil and Natural Gas": "ONGC.NS",
        "NTPC": "NTPC.NS",
        "Power Grid": "POWERGRID.NS",
        "Adani Green": "ADANIGREEN.NS",
        "Coal India": "COALINDIA.NS",
        "Tata Power": "TATAPOWER.NS",
        # Auto
        "Maruti": "MARUTI.NS", "Maruti Suzuki": "MARUTI.NS",
        "Tata Motors": "TATAMOTORS.NS",
        "Mahindra": "M&M.NS", "M&M": "M&M.NS",
        "Bajaj Auto": "BAJAJ-AUTO.NS",
        "Eicher": "EICHERMOT.NS", "Eicher Motors": "EICHERMOT.NS",
        "Hero Moto": "HEROMOTOCO.NS", "Hero MotoCorp": "HEROMOTOCO.NS",
        # FMCG
        "HUL": "HINDUNILVR.NS", "Hindustan Unilever": "HINDUNILVR.NS",
        "ITC": "ITC.NS",
        "Nestle": "NESTLEIND.NS", "Nestle India": "NESTLEIND.NS",
        "Britannia": "BRITANNIA.NS",
        "Tata Consumer": "TATACONSUM.NS",
        # Metals
        "Tata Steel": "TATASTEEL.NS",
        "JSW Steel": "JSWSTEEL.NS",
        "Hindalco": "HINDALCO.NS",
        "Adani Enterprises": "ADANIENT.NS", "Adani": "ADANIENT.NS",
        # Pharma
        "Sun Pharma": "SUNPHARMA.NS",
        "Dr Reddy": "DRREDDY.NS", "Dr Reddy's": "DRREDDY.NS",
        "Cipla": "CIPLA.NS",
        "Apollo Hospital": "APOLLOHOSP.NS", "Apollo": "APOLLOHOSP.NS",
        "Divi's Lab": "DIVISLAB.NS", "Divis": "DIVISLAB.NS",
        # Telecom
        "Bharti Airtel": "BHARTIARTL.NS", "Airtel": "BHARTIARTL.NS",
        # Infra & Cement
        "UltraTech": "ULTRACEMCO.NS", "UltraTech Cement": "ULTRACEMCO.NS",
        "Grasim": "GRASIM.NS",
        "Adani Ports": "ADANIPORTS.NS",
        "Shriram Finance": "SHRIRAMFIN.NS",
        "SBI Life": "SBILIFE.NS",
        "HDFC Life": "HDFCLIFE.NS",
        # Others
        "Titan": "TITAN.NS",
        "Asian Paints": "ASIANPAINT.NS",
        "Trent": "TRENT.NS",
        "BEL": "BEL.NS", "Bharat Electronics": "BEL.NS",
    })

    BENCHMARK: str = "^NSEI"  # Nifty 50 index
    VIX_TICKER: str = "^INDIAVIX"  # India Volatility Index


# ── Market Timing ───────────────────────────────────────────────
@dataclass(frozen=True)
class MarketConfig:
    """IST market hours and trading-day constraints."""

    TIMEZONE: str = "Asia/Kolkata"
    MARKET_OPEN: Tuple[int, int] = (9, 15)    # 9:15 AM IST
    MARKET_CLOSE: Tuple[int, int] = (15, 30)  # 3:30 PM IST
    NO_TRADE_END: Tuple[int, int] = (9, 30)   # No trades until 9:30 AM
    TRADING_DAYS: Tuple[int, ...] = (0, 1, 2, 3, 4)  # Mon–Fri


# ── Market Regime Thresholds ────────────────────────────────────
@dataclass(frozen=True)
class RegimeConfig:
    """Thresholds for the Market Regime Filter (crash protection)."""

    VIX_PANIC: float = 25.0       # VIX > 25 → PANIC mode, block ALL buys
    VIX_CAUTION: float = 20.0     # VIX > 20 → CAUTION mode, reduce position size
    NIFTY_MA_PERIOD: int = 50     # Nifty 50 SMA period for trend detection
    # Regime states: BULLISH, CAUTIOUS, BEARISH, PANIC


# ── Data Fetching & Retention ───────────────────────────────────
@dataclass(frozen=True)
class DataConfig:
    """Parameters for data fetching, rate-limiting, and RSS sources."""

    HISTORICAL_YEARS: int = 4       # Years of daily data to retain
    INTRADAY_DAYS: int = 59         # Max intraday window (yfinance cap ≈ 60d)
    FETCH_DELAY: float = 1.5        # Seconds between API calls (increased for 50 stocks)
    PRICE_REFRESH_MINS: int = 5     # Live price refresh cadence
    NEWS_REFRESH_MINS: int = 15     # News refresh cadence

    RSS_FEEDS: Tuple[Dict[str, str], ...] = (
        {"url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
         "source": "Economic Times"},
        {"url": "https://www.moneycontrol.com/rss/MCtopnews.xml",
         "source": "Moneycontrol"},
        {"url": "https://www.business-standard.com/rss/markets-106.rss",
         "source": "Business Standard"},
    )

    # yfinance maximum lookback per interval
    MAX_LOOKBACK: Dict[str, int] = field(default_factory=lambda: {
        "1m": 7, "2m": 60, "5m": 60, "15m": 60, "1h": 730,
    })


# ── ML Model Hyperparameters ───────────────────────────────────
@dataclass(frozen=True)
class ModelConfig:
    """Target thresholds, walk-forward CV settings, and model architectures."""

    # Target: price move 15 min ahead
    UP_THRESHOLD: float = 0.003     # > +0.3% → UP
    DOWN_THRESHOLD: float = -0.003  # < −0.3% → DOWN

    # Walk-forward validation
    WF_N_SPLITS: int = 5
    WF_TEST_DAYS: int = 5

    # XGBoost search grid
    XGB_PARAMS: Dict = field(default_factory=lambda: {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    })

    # Random Forest search grid
    RF_PARAMS: Dict = field(default_factory=lambda: {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [5, 10],
    })

    # LSTM architecture
    LSTM_SEQ_LEN: int = 20
    LSTM_UNITS: Tuple[int, int] = (64, 32)
    LSTM_DROPOUT: float = 0.2
    LSTM_EPOCHS: int = 100
    LSTM_BATCH_SIZE: int = 32
    LSTM_PATIENCE: int = 10

    RETRAIN_DAY: str = "sunday"


# ── Trading / Backtesting Constraints ───────────────────────────
@dataclass(frozen=True)
class TradingConfig:
    """Realistic brokerage, slippage, and position-sizing rules."""

    BROKERAGE_PCT: float = 0.001      # 0.1 % per trade
    SLIPPAGE_PCT: float = 0.0005      # 0.05 %
    MAX_POSITION_PCT: float = 0.20    # Max 20 % of portfolio per stock
    INITIAL_CAPITAL: float = 1_000_000.0  # ₹10 Lakh


# ── File Paths ──────────────────────────────────────────────────
@dataclass(frozen=True)
class PathConfig:
    """Derived paths for database, model artefacts, and logs."""

    DB_PATH: Path = BASE_DIR / "data" / "trading_assistant.db"
    MODEL_DIR: Path = BASE_DIR / "saved_models"
    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_FILE: Path = BASE_DIR / "logs" / "trading_assistant.log"


# ── Singleton Instances (import these) ──────────────────────────
STOCKS  = StockConfig()
MARKET  = MarketConfig()
REGIME  = RegimeConfig()
DATA    = DataConfig()
MODEL   = ModelConfig()
TRADING = TradingConfig()
PATHS   = PathConfig()
