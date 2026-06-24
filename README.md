# Stonks

An automated, AI-powered trading assistant built for the Indian Stock Market (Nifty 50). This bot fetches historical and real-time market data, calculates advanced technical and Smart Money Concepts (SMC) indicators, performs news sentiment analysis using FinBERT, and generates predictive BUY/SELL signals using Machine Learning (XGBoost/Random Forest). 

It features a Telegram bot integration to deliver real-time trade alerts, chart generation, and market regime tracking.

## Features

- **Nifty 50 Tracking**: Automatically tracks and analyzes a predefined list of Nifty 50 stocks.
- **Machine Learning Signals**: Uses trained ML models (XGBoost / Random Forest) to predict market movements and generate confidence-backed trade signals.
- **Smart Money Concepts (SMC) & Technicals**: Computes Fair Value Gaps (FVG), Order Blocks (OB), Liquidity Sweeps, MACD, RSI, ATR, and Volume Spikes.
- **Market Regime Detection**: Identifies whether the market is Bullish, Cautious, Bearish, or in Panic. Automatically blocks BUY signals during Bearish/Panic regimes to protect capital.
- **News Sentiment Analysis**: Periodically fetches financial news and scores them using FinBERT.
- **Telegram Bot Integration**: Sends detailed trade setups (Entry, Take Profit, Stop Loss) and supports interactive commands (`/chart`, `/status`, `/help`).
- **Fully Automated Pipeline**: Uses `APScheduler` to refresh prices every 5 minutes, news every 15 minutes, and retrains the model automatically every week.

## Setup & Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Stonk
```

### 2. Activate the Virtual Environment
Before running or installing anything, activate the virtual environment:

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```
*(If you are using Command Prompt, use `.venv\Scripts\activate.bat`)*

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies
Once the environment is activated `(.venv)`, install the required packages:
```bash
pip install -r trading_assistant/requirements.txt
```

### 4. Configuration
Ensure you have set up your `trading_assistant/config.py` and provided any required API keys (e.g., Telegram Bot Token) in your environment variables or `.env` file.

## Usage

To start the trading assistant pipeline, ensure your virtual environment is active and run:

```bash
python -m trading_assistant.main
```

### What happens on startup?
1. **Initial Data Load**: Fetches up to 4 years of daily data and 60 days of 15-minute data.
2. **News & Sentiment**: Fetches the latest news and runs sentiment analysis.
3. **Model Loading/Training**: Loads the existing ML model or trains a new one if none exists.
4. **Market Regime Check**: Evaluates the current market conditions.
5. **Live Scheduler**: Starts tracking the market and sending alerts to Telegram.

## Telegram Commands
Once the bot is running, you can interact with it on Telegram:
- `/chart <TICKER>` - View the latest chart for a specific stock (e.g., `/chart RELIANCE.NS`)
- `/status` - Check the current Market Regime
- `/help` - View all available commands

## Project Structure
- `main.py`: The main entry point and scheduler.
- `config.py`: Configuration settings for data, models, and stocks.
- `data/`: Modules for fetching price data, news, and managing the database.
- `features/`: Technical indicators, SMC logic, market regime detection, and sentiment analysis.
- `models/`: Machine learning training pipeline and prediction logic.
- `notifications.py`: Telegram bot integration and message formatting.
