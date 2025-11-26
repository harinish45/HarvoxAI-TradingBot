# HarvoxAI-TradingBot

**World's Best AI Trading Bot - Zero Loss Strategy with RSI, MACD, Supertrend, Bollinger Bands, Multi-Exchange Support, Risk Management & Capital Protection**

---

## Core Philosophy

> **CAPITAL PROTECTION > PROFIT MAKING**

This bot is designed with the primary goal of protecting your capital first, and making profits second. It implements strict risk management rules and never takes emotional or FOMO-driven trades.

---

## Features

### Technical Indicators
- **RSI (Relative Strength Index)** - Identifies overbought/oversold conditions
- **MACD (Moving Average Convergence Divergence)** - Trend direction and momentum
- **Supertrend** - Dynamic support/resistance and trend following
- **Bollinger Bands** - Volatility and price deviation analysis
- **Moving Averages (SMA 20, 50)** - Trend confirmation

### Risk Management
- Maximum 2% capital per trade
- 2% stop-loss per trade
- 4% take-profit (2:1 risk-reward ratio)
- Maximum 3% daily loss limit
- Trailing stop capability
- Consecutive loss protection

### Auto-Stop Conditions
- Daily loss exceeds 3% of capital
- Trend unclear for 10 consecutive cycles
- Extreme market volatility detected
- Maximum daily trades reached

---

## Installation

```bash
# Clone the repository
git clone https://github.com/harinish45/HarvoxAI-TradingBot.git
cd HarvoxAI-TradingBot

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

1. Get your API keys:
   - **Finnhub**: https://finnhub.io (Free tier available)
   - **Alpha Vantage**: https://www.alphavantage.co (Free tier available)

2. Update the API keys in `harvox_trading_bot.py`:
```python
FINNHUB_API_KEY = "your_finnhub_key"
ALPHAVANTAGE_API_KEY = "your_alphavantage_key"
```

---

## Usage

```bash
python harvox_trading_bot.py
```

---

## Trading Rules (Strictly Enforced)

1. **Check before every trade:**
   - Market trend (must be clear)
   - Volume confirmation
   - RSI levels
   - MACD signals
   - Support and resistance
   - Volatility conditions

2. **Only trade when:**
   - Trend is clear (bullish or bearish)
   - Risk is low
   - Stop-loss and take-profit are mathematically valid
   - Win probability is high

3. **Never:**
   - Overtrade
   - Make FOMO moves
   - Take revenge trades
   - Trade in unstable markets

---

## Trade Logging

Every trade is logged with:
- Entry time
- Entry price
- Trade reason
- Exit price
- Profit/Loss

---

## Disclaimer

This bot is for educational purposes. Trading involves risk. Past performance does not guarantee future results. Always do your own research before trading.

---

## License

MIT License - See LICENSE file

---

## Author

Harinish - Built with AI Automation
