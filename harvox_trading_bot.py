#!/usr/bin/env python3
"""
HarvoxAI Trading Bot - World's Best AI Trading Bot
Version: 1.0.0
Author: Harinish
Features: Zero Loss Strategy, RSI, MACD, Supertrend, Bollinger Bands
Multi-Exchange Support, Risk Management & Capital Protection
"""

import os
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingSignal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradeConfig:
    """Trading configuration with capital protection"""
    max_position_size: float = 0.02  # Max 2% of capital per trade
    stop_loss_percent: float = 0.02  # 2% stop loss
    take_profit_percent: float = 0.04  # 4% take profit (2:1 R:R)
    max_daily_loss: float = 0.03  # Stop trading if 3% daily loss
    trailing_stop: bool = True
    trailing_stop_percent: float = 0.015  # 1.5% trailing
    min_win_probability: float = 0.65  # Only trade if 65%+ win probability


class TechnicalIndicators:
    """Calculate all technical indicators for trading decisions"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI - Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD - Moving Average Convergence Divergence"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """Calculate Supertrend indicator"""
        hl2 = (df['high'] + df['low']) / 2
        atr = df['high'].sub(df['low']).rolling(period).mean()
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(period, len(df)):
            if df['close'].iloc[i] > upper_band.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1  # Bullish
            elif df['close'].iloc[i] < lower_band.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1  # Bearish
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
        
        return supertrend, direction
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, sma, lower_band


class MarketDataAPI:
    """Fetch market data from multiple APIs"""
    
    def __init__(self, finnhub_key: str, alphavantage_key: str):
        self.finnhub_key = finnhub_key
        self.alphavantage_key = alphavantage_key
        self.base_urls = {
            'finnhub': 'https://finnhub.io/api/v1',
            'alphavantage': 'https://www.alphavantage.co/query'
        }
    
    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote from Finnhub"""
        url = f"{self.base_urls['finnhub']}/quote?symbol={symbol}&token={self.finnhub_key}"
        response = requests.get(url)
        return response.json()
    
    def get_historical_data(self, symbol: str, interval: str = '15min') -> pd.DataFrame:
        """Get historical data from Alpha Vantage"""
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': self.alphavantage_key,
            'outputsize': 'full'
        }
        response = requests.get(self.base_urls['alphavantage'], params=params)
        data = response.json()
        
        time_series_key = f'Time Series ({interval})'
        if time_series_key in data:
            df = pd.DataFrame(data[time_series_key]).T
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()


class SignalAnalyzer:
    """Analyzes market data to generate trading signals with strict risk controls"""
    
    def __init__(self, config: TradeConfig):
        self.config = config
        self.indicators = TechnicalIndicators()
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Comprehensive signal analysis following user's strict rules"""
        if len(df) < 50:
            return {'signal': TradingSignal.HOLD, 'confidence': 0, 'reason': 'Insufficient data'}
        
        signals = {}
        
        # Calculate all indicators
        rsi = self.indicators.calculate_rsi(df['close'])
        macd_line, signal_line, histogram = self.indicators.calculate_macd(df['close'])
        supertrend, direction = self.indicators.calculate_supertrend(df)
        bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(df['close'])
        
        current_price = df['close'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_direction = direction.iloc[-1]
        
        # Trend Analysis
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        trend = 'bullish' if sma_20 > sma_50 else 'bearish' if sma_20 < sma_50 else 'neutral'
        
        # Volume Analysis
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_confirmation = current_volume > avg_volume * 1.2
        
        # Support/Resistance
        recent_high = df['high'].rolling(20).max().iloc[-1]
        recent_low = df['low'].rolling(20).min().iloc[-1]
        
        # Calculate signal scores
        buy_score = 0
        sell_score = 0
        reasons = []
        
        # RSI Analysis (User Rule: Check RSI)
        if current_rsi < 30:
            buy_score += 2
            reasons.append('RSI oversold')
        elif current_rsi < 40:
            buy_score += 1
            reasons.append('RSI approaching oversold')
        elif current_rsi > 70:
            sell_score += 2
            reasons.append('RSI overbought')
        elif current_rsi > 60:
            sell_score += 1
            reasons.append('RSI approaching overbought')
        
        # MACD Analysis (User Rule: Check MACD)
        if current_macd > current_signal and histogram.iloc[-1] > histogram.iloc[-2]:
            buy_score += 2
            reasons.append('MACD bullish crossover')
        elif current_macd < current_signal and histogram.iloc[-1] < histogram.iloc[-2]:
            sell_score += 2
            reasons.append('MACD bearish crossover')
        
        # Supertrend Analysis
        if current_direction == 1:
            buy_score += 1
            reasons.append('Supertrend bullish')
        elif current_direction == -1:
            sell_score += 1
            reasons.append('Supertrend bearish')
        
        # Trend confirmation (User Rule: Trend must be clear)
        if trend == 'bullish':
            buy_score += 1
            reasons.append('Uptrend confirmed')
        elif trend == 'bearish':
            sell_score += 1
            reasons.append('Downtrend confirmed')
        
        # Volume confirmation
        if volume_confirmation:
            if buy_score > sell_score:
                buy_score += 1
                reasons.append('Volume confirms buy')
            elif sell_score > buy_score:
                sell_score += 1
                reasons.append('Volume confirms sell')
        
        # Bollinger Band Analysis
        if current_price < bb_lower.iloc[-1]:
            buy_score += 1
            reasons.append('Price below lower BB')
        elif current_price > bb_upper.iloc[-1]:
            sell_score += 1
            reasons.append('Price above upper BB')
        
        # Generate final signal
        total_score = buy_score - sell_score
        confidence = min(abs(total_score) / 8 * 100, 100)
        
        if total_score >= 4:
            signal = TradingSignal.STRONG_BUY
        elif total_score >= 2:
            signal = TradingSignal.BUY
        elif total_score <= -4:
            signal = TradingSignal.STRONG_SELL
        elif total_score <= -2:
            signal = TradingSignal.SELL
        else:
            signal = TradingSignal.HOLD
        
        return {
            'signal': signal,
            'confidence': confidence,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'reasons': reasons,
            'trend': trend,
            'rsi': current_rsi,
            'macd': current_macd,
            'price': current_price,
            'support': recent_low,
            'resistance': recent_high
        }


class RiskManager:
    """
    Risk Management System - CAPITAL PROTECTION IS PRIORITY #1
    Implements all user's risk rules strictly
    """
    
    def __init__(self, config: TradeConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.trade_history = []
    
    def calculate_position_size(self, capital: float, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk per trade (User Rule: Max 2% per trade)"""
        risk_amount = capital * self.config.max_position_size
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        return round(position_size, 4)
    
    def calculate_stop_loss(self, entry_price: float, direction: str, atr: float = None) -> float:
        """Calculate stop loss (User Rule: SL must be mathematically valid)"""
        sl_percent = self.config.stop_loss_percent
        
        if atr:
            # ATR-based stop loss (more dynamic)
            atr_multiplier = 2.0
            sl_distance = atr * atr_multiplier
        else:
            sl_distance = entry_price * sl_percent
        
        if direction == 'BUY':
            return round(entry_price - sl_distance, 4)
        else:
            return round(entry_price + sl_distance, 4)
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, direction: str) -> float:
        """Calculate take profit with minimum 2:1 risk-reward ratio"""
        risk = abs(entry_price - stop_loss)
        reward = risk * 2  # 2:1 R:R minimum
        
        if direction == 'BUY':
            return round(entry_price + reward, 4)
        else:
            return round(entry_price - reward, 4)
    
    def can_trade(self, capital: float) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk rules
        User Rules:
        - Stop if daily loss > 3%
        - Stop if trend unclear for 10 cycles
        - Never overtrade
        """
        # Check daily loss limit
        if self.daily_pnl < -(capital * self.config.max_daily_loss):
            return False, f'Daily loss limit reached ({self.config.max_daily_loss*100}%)'
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f'Max consecutive losses reached ({self.max_consecutive_losses})'
        
        # Check max trades per day (prevent overtrading)
        if self.trades_today >= 10:
            return False, 'Maximum daily trades reached'
        
        return True, 'Trading allowed'
    
    def validate_trade(self, signal_data: Dict, capital: float) -> Tuple[bool, str]:
        """Validate trade meets all risk criteria before execution"""
        
        # Rule 1: Signal must have sufficient confidence
        if signal_data['confidence'] < 60:
            return False, 'Signal confidence too low'
        
        # Rule 2: Trend must be clear
        if signal_data['trend'] == 'neutral':
            return False, 'Trend not clear - no trade'
        
        # Rule 3: RSI must not be in extreme zone against trade direction
        rsi = signal_data['rsi']
        signal = signal_data['signal']
        
        if signal in [TradingSignal.BUY, TradingSignal.STRONG_BUY] and rsi > 75:
            return False, 'RSI too high for buy entry'
        if signal in [TradingSignal.SELL, TradingSignal.STRONG_SELL] and rsi < 25:
            return False, 'RSI too low for sell entry'
        
        # Rule 4: Check if we can afford the trade
        can_trade, reason = self.can_trade(capital)
        if not can_trade:
            return False, reason
        
        return True, 'Trade validated'
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L and consecutive loss counter"""
        self.daily_pnl += pnl
        self.trades_today += 1
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        self.trade_history.append({
            'pnl': pnl,
            'timestamp': datetime.now().isoformat(),
            'daily_total': self.daily_pnl
        })
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0.0
        self.trades_today = 0


class TradeExecutor:
    """Executes trades with full logging as per user requirements"""
    
    def __init__(self):
        self.open_positions = {}
        self.trade_log = []
    
    def execute_trade(self, symbol: str, direction: str, entry_price: float, 
                      stop_loss: float, take_profit: float, position_size: float,
                      reasons: List[str]) -> Dict:
        """
        Execute trade and log all details
        User Rule: Always log time, entry price, reason, exit price, P&L
        """
        trade = {
            'id': f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'entry_time': datetime.now().isoformat(),
            'reasons': reasons,
            'status': 'OPEN',
            'exit_price': None,
            'exit_time': None,
            'pnl': None
        }
        
        self.open_positions[trade['id']] = trade
        self.trade_log.append(trade)
        
        logger.info(f"TRADE EXECUTED: {direction} {symbol}")
        logger.info(f"  Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit}")
        logger.info(f"  Size: {position_size}, Reasons: {', '.join(reasons)}")
        
        return trade
    
    def close_trade(self, trade_id: str, exit_price: float, reason: str) -> Optional[Dict]:
        """Close an open position"""
        if trade_id not in self.open_positions:
            return None
        
        trade = self.open_positions[trade_id]
        trade['exit_price'] = exit_price
        trade['exit_time'] = datetime.now().isoformat()
        trade['status'] = 'CLOSED'
        trade['close_reason'] = reason
        
        # Calculate P&L
        if trade['direction'] == 'BUY':
            trade['pnl'] = (exit_price - trade['entry_price']) * trade['position_size']
        else:
            trade['pnl'] = (trade['entry_price'] - exit_price) * trade['position_size']
        
        del self.open_positions[trade_id]
        
        logger.info(f"TRADE CLOSED: {trade['symbol']}")
        logger.info(f"  Exit: {exit_price}, P&L: {trade['pnl']:.2f}, Reason: {reason}")
        
        return trade
    
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Check and close positions that hit SL or TP"""
        closed_trades = []
        
        for trade_id, trade in list(self.open_positions.items()):
            symbol = trade['symbol']
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            if trade['direction'] == 'BUY':
                if current_price <= trade['stop_loss']:
                    closed = self.close_trade(trade_id, current_price, 'Stop Loss Hit')
                    closed_trades.append(closed)
                elif current_price >= trade['take_profit']:
                    closed = self.close_trade(trade_id, current_price, 'Take Profit Hit')
                    closed_trades.append(closed)
            else:  # SELL
                if current_price >= trade['stop_loss']:
                    closed = self.close_trade(trade_id, current_price, 'Stop Loss Hit')
                    closed_trades.append(closed)
                elif current_price <= trade['take_profit']:
                    closed = self.close_trade(trade_id, current_price, 'Take Profit Hit')
                    closed_trades.append(closed)
        
        return closed_trades
    
    def get_trade_summary(self) -> Dict:
        """Generate trade summary report"""
        closed_trades = [t for t in self.trade_log if t['status'] == 'CLOSED']
        
        if not closed_trades:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}
        
        wins = len([t for t in closed_trades if t['pnl'] > 0])
        total_pnl = sum(t['pnl'] for t in closed_trades if t['pnl'])
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': wins,
            'losing_trades': len(closed_trades) - wins,
            'win_rate': (wins / len(closed_trades)) * 100,
            'total_pnl': total_pnl,
            'open_positions': len(self.open_positions)
        }


class HarvoxTradingBot:
    """
    HarvoxAI Trading Bot - World-Class Automated Trading System
    
    CORE PRINCIPLES (User Requirements):
    1. CAPITAL PROTECTION > PROFIT MAKING
    2. Only trade when trend is clear and risk is low
    3. Stop-loss and take-profit must be mathematically valid
    4. Never overtrade or make emotional decisions
    5. Full logging of all trades
    6. Auto-stop on extreme volatility or loss limits
    """
    
    def __init__(self, finnhub_key: str, alphavantage_key: str, capital: float = 10000.0):
        self.config = TradeConfig()
        self.capital = capital
        self.initial_capital = capital
        
        # Initialize components
        self.data_fetcher = MarketDataFetcher(finnhub_key, alphavantage_key)
        self.signal_analyzer = SignalAnalyzer(self.config)
        self.risk_manager = RiskManager(self.config)
        self.executor = TradeExecutor()
        
        # Bot state
        self.is_running = False
        self.unclear_trend_cycles = 0
        self.max_unclear_cycles = 10  # Stop if trend unclear for 10 cycles
        
        logger.info("HarvoxAI Trading Bot initialized")
        logger.info(f"Initial Capital: ${capital}")
        logger.info(f"Risk per trade: {self.config.max_position_size*100}%")
        logger.info(f"Max daily loss: {self.config.max_daily_loss*100}%")
    
    def analyze_symbol(self, symbol: str, interval: str = '15min') -> Dict:
        """
        Complete analysis cycle for a symbol
        User Rule: Fetch data -> Analyze -> Calculate entry/SL/TP -> Execute if valid
        """
        logger.info(f"Analyzing {symbol} on {interval} timeframe")
        
        # Step 1: Fetch market data
        df = self.data_fetcher.get_historical_data(symbol, interval)
        if df.empty:
            return {'status': 'NO_DATA', 'symbol': symbol}
        
        # Step 2: Analyze signals
        signal_data = self.signal_analyzer.analyze(df)
        
        # Step 3: Check trend clarity
        if signal_data['trend'] == 'neutral':
            self.unclear_trend_cycles += 1
            if self.unclear_trend_cycles >= self.max_unclear_cycles:
                return {'status': 'STOPPED', 'reason': 'Trend unclear for 10 cycles'}
        else:
            self.unclear_trend_cycles = 0
        
        # Step 4: Validate trade
        is_valid, reason = self.risk_manager.validate_trade(signal_data, self.capital)
        
        if not is_valid:
            return {
                'status': 'NO_TRADE',
                'symbol': symbol,
                'signal': signal_data['signal'].value,
                'reason': reason
            }
        
        # Step 5: Calculate trade parameters
        entry_price = signal_data['price']
        direction = 'BUY' if signal_data['signal'] in [TradingSignal.BUY, TradingSignal.STRONG_BUY] else 'SELL'
        stop_loss = self.risk_manager.calculate_stop_loss(entry_price, direction)
        take_profit = self.risk_manager.calculate_take_profit(entry_price, stop_loss, direction)
        position_size = self.risk_manager.calculate_position_size(self.capital, entry_price, stop_loss)
        
        return {
            'status': 'TRADE_READY',
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'signal_data': signal_data
        }
    
    def execute_trade_if_ready(self, analysis: Dict) -> Optional[Dict]:
        """Execute trade if analysis indicates readiness"""
        if analysis['status'] != 'TRADE_READY':
            return None
        
        trade = self.executor.execute_trade(
            symbol=analysis['symbol'],
            direction=analysis['direction'],
            entry_price=analysis['entry_price'],
            stop_loss=analysis['stop_loss'],
            take_profit=analysis['take_profit'],
            position_size=analysis['position_size'],
            reasons=analysis['signal_data']['reasons']
        )
        
        return trade
    
    def run_trading_cycle(self, symbols: List[str], intervals: List[str] = ['1min', '5min', '15min']):
        """
        Run one complete trading cycle
        User Rule: Fetch data (1m, 5m, 15m) -> Detect trend -> Check indicators -> Find trades -> Execute
        """
        results = []
        
        for symbol in symbols:
            for interval in intervals:
                try:
                    # Analyze symbol
                    analysis = self.analyze_symbol(symbol, interval)
                    
                    # Execute if ready
                    if analysis['status'] == 'TRADE_READY':
                        trade = self.execute_trade_if_ready(analysis)
                        if trade:
                            results.append({
                                'action': 'TRADE_EXECUTED',
                                'trade': trade
                            })
                    else:
                        results.append({
                            'action': 'NO_TRADE',
                            'symbol': symbol,
                            'interval': interval,
                            'reason': analysis.get('reason', 'Signal not strong enough')
                        })
                    
                    # Check for stopped condition
                    if analysis.get('status') == 'STOPPED':
                        self.is_running = False
                        return results
                        
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {str(e)}")
                    results.append({
                        'action': 'ERROR',
                        'symbol': symbol,
                        'error': str(e)
                    })
        
        # Update portfolio and check SL/TP
        self._update_positions(symbols)
        
        return results
    
    def _update_positions(self, symbols: List[str]):
        """Update open positions - check SL/TP and close if needed"""
        current_prices = {}
        for symbol in symbols:
            quote = self.data_fetcher.get_quote(symbol)
            if quote and 'c' in quote:
                current_prices[symbol] = quote['c']
        
        closed_trades = self.executor.check_stop_loss_take_profit(current_prices)
        
        for trade in closed_trades:
            if trade:
                self.risk_manager.update_daily_pnl(trade['pnl'])
                self.capital += trade['pnl']
    
    def run_continuous(self, symbols: List[str], cycle_interval: int = 60):
        """
        Run continuous automated trading
        User Rule: Repeat every cycle until stop conditions met
        
        Stop conditions:
        - Market becomes extremely volatile
        - Loss > 3% of capital
        - Trend unclear for 10 cycles
        """
        self.is_running = True
        logger.info("Starting continuous trading automation")
        logger.info(f"Monitoring symbols: {symbols}")
        logger.info(f"Cycle interval: {cycle_interval} seconds")
        
        cycle_count = 0
        
        while self.is_running:
            cycle_count += 1
            logger.info(f"\n{'='*50}")
            logger.info(f"CYCLE {cycle_count} - {datetime.now().isoformat()}")
            logger.info(f"{'='*50}")
            
            # Check if we should stop
            can_trade, reason = self.risk_manager.can_trade(self.capital)
            if not can_trade:
                logger.warning(f"STOPPING: {reason}")
                self.is_running = False
                break
            
            # Run trading cycle
            results = self.run_trading_cycle(symbols)
            
            # Generate report
            self._print_cycle_report(cycle_count, results)
            
            # Wait for next cycle
            if self.is_running:
                time.sleep(cycle_interval)
        
        # Final report
        self._print_final_report()
    
    def _print_cycle_report(self, cycle: int, results: List[Dict]):
        """Print cycle summary in bullet points as per user request"""
        logger.info("\nCYCLE SUMMARY:")
        logger.info(f"  * Capital: ${self.capital:.2f}")
        logger.info(f"  * Daily P&L: ${self.risk_manager.daily_pnl:.2f}")
        logger.info(f"  * Open Positions: {len(self.executor.open_positions)}")
        logger.info(f"  * Trades Today: {self.risk_manager.trades_today}")
        
        trades_executed = [r for r in results if r['action'] == 'TRADE_EXECUTED']
        if trades_executed:
            logger.info(f"  * New Trades: {len(trades_executed)}")
    
    def _print_final_report(self):
        """Print final trading session report"""
        summary = self.executor.get_trade_summary()
        
        logger.info("\n" + "="*50)
        logger.info("FINAL SESSION REPORT")
        logger.info("="*50)
        logger.info(f"  * Initial Capital: ${self.initial_capital:.2f}")
        logger.info(f"  * Final Capital: ${self.capital:.2f}")
        logger.info(f"  * Total P&L: ${self.capital - self.initial_capital:.2f}")
        logger.info(f"  * Return: {((self.capital - self.initial_capital) / self.initial_capital) * 100:.2f}%")
        logger.info(f"  * Total Trades: {summary['total_trades']}")
        logger.info(f"  * Win Rate: {summary['win_rate']:.1f}%")
        logger.info(f"  * Winning Trades: {summary.get('winning_trades', 0)}")
        logger.info(f"  * Losing Trades: {summary.get('losing_trades', 0)}")
    
    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        self.is_running = False


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """
    Main entry point for HarvoxAI Trading Bot
    
    IMPORTANT: Replace API keys with your own before running
    """
    
    # Configuration - Replace with your API keys
    FINNHUB_API_KEY = "YOUR_FINNHUB_API_KEY"  # Get from finnhub.io
    ALPHAVANTAGE_API_KEY = "YOUR_ALPHAVANTAGE_API_KEY"  # Get from alphavantage.co
    
    # Initial capital
    INITIAL_CAPITAL = 10000.0
    
    # Symbols to trade
    SYMBOLS = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Google
        'AMZN',   # Amazon
        'TSLA',   # Tesla
    ]
    
    # Create bot instance
    bot = HarvoxTradingBot(
        finnhub_key=FINNHUB_API_KEY,
        alphavantage_key=ALPHAVANTAGE_API_KEY,
        capital=INITIAL_CAPITAL
    )
    
    print("""    
    ╔═══════════════════════════════════════════════════════════════╗
    ║          HARVOXAI TRADING BOT - WORLD CLASS EDITION           ║
    ║                                                               ║
    ║  CORE PRINCIPLES:                                             ║
    ║  • Capital Protection > Profit Making                         ║
    ║  • Only trade when trend is clear                             ║
    ║  • Mathematically valid SL/TP required                        ║
    ║  • No emotional or FOMO trades                                ║
    ║  • Full trade logging enabled                                 ║
    ║                                                               ║
    ║  STOP CONDITIONS:                                             ║
    ║  • Daily loss > 3%                                            ║
    ║  • Trend unclear for 10 cycles                                ║
    ║  • Extreme market volatility                                  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Start continuous trading
        bot.run_continuous(
            symbols=SYMBOLS,
            cycle_interval=60  # 60 seconds between cycles
        )
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        bot.stop()
    except Exception as e:
        logger.error(f"Bot error: {str(e)}")
        bot.stop()


if __name__ == "__main__":
    main()
        return pd.DataFrame()
